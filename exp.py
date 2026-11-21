"""python run_experiments.py [--seeds 5 --rounds 40 --clients 5 --frac 0.6 --alpha 0.5]  ->  results/*.csv, meta.json"""
import argparse, io, json, os
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import confusion_matrix
import crypto_envelope as ce, data_diabetes as D

p = argparse.ArgumentParser()
p.add_argument("--seeds", type=int, default=5); p.add_argument("--rounds", type=int, default=40)
p.add_argument("--clients", type=int, default=5); p.add_argument("--frac", type=float, default=0.6)
p.add_argument("--epochs", type=int, default=2); p.add_argument("--lr", type=float, default=0.05)
p.add_argument("--bs", type=int, default=16); p.add_argument("--alpha", type=float, default=0.5)
p.add_argument("--eps", type=float, default=0.005); p.add_argument("--W", type=int, default=3)
p.add_argument("--out", default="results")
a = p.parse_args(); os.makedirs(a.out, exist_ok=True); torch.set_num_threads(1)


class MLP(nn.Module):  # 8 -> 64 -> 32 -> 1, ReLU, dropout
    def __init__(s):
        super().__init__()
        s.n = nn.Sequential(nn.Linear(8, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(s, x): return s.n(x).squeeze(1)


get_w = lambda m: [v.detach().numpy().copy() for v in m.state_dict().values()]
def set_w(m, w): m.load_state_dict({k: torch.tensor(v) for k, v in zip(m.state_dict().keys(), w)})
def ser(w): b = io.BytesIO(); np.savez(b, *w); return b.getvalue()
def deser(b): z = np.load(io.BytesIO(b)); return [z[f"arr_{i}"] for i in range(len(z.files))]


def local_train(m, X, y, rng):
    m.train(); opt = torch.optim.SGD(m.parameters(), lr=a.lr); lf = nn.BCEWithLogitsLoss()
    for _ in range(a.epochs):
        perm = rng.permutation(len(X))
        for i in range(0, len(X), a.bs):
            b = perm[i:i + a.bs]; opt.zero_grad(); lf(m(torch.from_numpy(X[b])), torch.from_numpy(y[b])).backward(); opt.step()


@torch.no_grad()
def evaluate(m, X, y):
    m.eval(); z = m(torch.from_numpy(X)); loss = float(nn.BCEWithLogitsLoss()(z, torch.from_numpy(y)))
    tn, fp, fn, tp = confusion_matrix(y.astype(int), (z > 0).numpy().astype(int), labels=[0, 1]).ravel()
    pr = tp / (tp + fp) if tp + fp else 0.0; rc = tp / (tp + fn) if tp + fn else 0.0
    return (tp + tn) / len(y), pr, rc, (2 * pr * rc / (pr + rc) if pr + rc else 0.0), loss


def run(seed, mode, dist, frac, parts, Xtr, ytr, Xte, yte, A, O):
    torch.manual_seed(seed); rng = np.random.default_rng(seed + 1000)
    gm, local, env = MLP(), MLP(), ce.Envelope(mode); w = get_w(gm); K = len(parts); m = max(1, int(round(frac * K)))
    for t in range(1, a.rounds + 1):
        env.new_round(); S = rng.choice(K, m, replace=False); recv, ns = [], []
        for k in S:
            set_w(local, w); local_train(local, Xtr[parts[k]], ytr[parts[k]], rng)
            pay = ser(get_w(local)); out, st = env.transmit(pay); assert out == pay and st.ok
            recv.append(deser(out)); ns.append(len(parts[k]))
            O.append(dict(seed=seed, envelope=mode, dist=dist, frac=frac, round=t, kex_up=st.kex_up_bytes, kex_down=st.kex_down_bytes,
                          aead_oh=st.aead_overhead, payload=st.payload_bytes, t_kex=st.t_kex_ms, t_kdf=st.t_kdf_ms, t_aead=st.t_aead_ms, total=st.total_ms))
        tot = float(sum(ns)); w = [sum(n / tot * r[i] for n, r in zip(ns, recv)) for i in range(len(w))]; set_w(gm, w)
        ac, pr, rc, f1, lo = evaluate(gm, Xte, yte)
        A.append(dict(seed=seed, envelope=mode, dist=dist, frac=frac, round=t, acc=ac, precision=pr, recall=rc, f1=f1, loss=lo))


df = D.load_raw(); A, O, skew = [], [], []
for seed in range(a.seeds):
    Xtr, ytr, Xte, yte = D.make_split(df, seed); rng = np.random.default_rng(seed)
    iid = D.partition_iid(len(Xtr), a.clients, rng); non = D.partition_dirichlet(ytr, a.clients, a.alpha, rng)
    skew.append(dict(seed=seed, iid_pos=[float(ytr[q].mean()) for q in iid], non_pos=[float(ytr[q].mean()) for q in non],
                     iid_n=[len(q) for q in iid], non_n=[len(q) for q in non]))
    for dist, parts in (("iid", iid), ("noniid", non)):
        for mode in ("none", "classical", "pqc"): run(seed, mode, dist, a.frac, parts, Xtr, ytr, Xte, yte, A, O)
    for fr in (0.4, 1.0): run(seed, "pqc", "noniid", fr, non, Xtr, ytr, Xte, yte, A, O)
    print("seed", seed, "done", flush=True)
pd.DataFrame(A).to_csv(f"{a.out}/accuracy.csv", index=False); pd.DataFrame(O).to_csv(f"{a.out}/overhead.csv", index=False)
json.dump(dict(args=vars(a), n_train=int(len(Xtr)), n_test=int(len(Xte)), prevalence=float(df.outcome.mean()), skew=skew,
               backend=ce.KEM_BACKEND, nparam=int(sum(q.numel() for q in MLP().parameters())), torch=torch.__version__),
          open(f"{a.out}/meta.json", "w"), indent=1)
print("done")