"""Fit the residual radial distortion left by the lensfun profile together with the sky rotation.
Model (u = lensfun-undistorted coords, d = final coords):
    d = D(u):  r_d = r (1 + k1 rho^2 + k2 rho^4),  rho = r / R0, about the frame centre
    D(p_ref) = H_i D(p_i),  H_i = K R(theta_i about the pole axis) K^-1
Unknowns: k1, k2, f, pole (x, y), theta_i per frame. Pairs come from the star lists of every frame.
Writes model.npz: k1, k2, f, pole, H (90 x 3 x 3 in d-space, identity for the reference).
"""
import numpy as np, cv2
from multiprocessing import Pool
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
import register as R

W = R.W
R0 = np.hypot(R.CX, R.CY)
N = R.P.N


def D(p, k1, k2):
    x, y = p[:, 0] - R.CX, p[:, 1] - R.CY
    rho2 = (x * x + y * y) / R0 ** 2
    s = 1 + k1 * rho2 + k2 * rho2 * rho2
    return np.c_[R.CX + x * s, R.CY + y * s]


def Hmat(theta, f, pole):
    k = np.array([[f, 0, R.CX], [0, f, R.CY], [0, 0, 1.0]])
    a = np.linalg.inv(k) @ np.array([pole[0], pole[1], 1.0]); a /= np.linalg.norm(a)
    Rm, _ = cv2.Rodrigues(a * theta)
    return k @ Rm @ np.linalg.inv(k)


def residuals(x, pairs, idx):
    k1, k2, f, px, py = x[:5]
    thetas = x[5:]
    out = []
    for j, (pr, pc) in enumerate(pairs):
        i = idx[j]
        dr = D(pr, k1, k2); dc = D(pc, k1, k2)
        H = Hmat(thetas[j], f, (px, py)) if i != R.REF else np.eye(3)
        out.append((dr - R.apply(H, dc)).ravel())
    return np.concatenate(out)


def main():
    times = R.frame_times()
    Hs = np.load(f"{W}/H_all.npy", allow_pickle=True).item()
    with Pool(6) as p:
        stars = dict(zip(range(1, N + 1), p.map(R.detect, range(1, N + 1))))
    np.savez(f"{W}/stars.npz", **{str(i): s for i, s in stars.items()})
    ref = stars[R.REF]
    tree = cKDTree(ref[:, :2])
    pairs, idx = [], []
    for i in range(1, N + 1):
        if i == R.REF or i not in Hs:
            continue
        cur = stars[i]
        pred = R.apply(Hs[i], cur)
        d, j = tree.query(pred, distance_upper_bound=6)
        ok = np.isfinite(d)
        if ok.sum() < 8:
            continue
        pairs.append((ref[j[ok], :2], cur[ok, :2])); idx.append(i)
    print("frames in fit", len(idx), "pairs", sum(len(a) for a, _ in pairs), flush=True)
    x0 = np.r_[0.0, 0.0, R.F_PX, 1700.0, R.HEIGHT - 717.0, [-R.OMEGA * times[i] for i in idx]]
    for tol in (6, 3):
        lo = np.r_[-0.2, -0.2, 0.3 * R.F_PX, -np.inf, -np.inf, np.full(len(idx), -np.inf)]     # residual distortion stays small
        hi = np.r_[0.2, 0.2, 3.0 * R.F_PX, np.inf, np.inf, np.full(len(idx), np.inf)]
        x0 = np.clip(x0, lo + 1e-9, hi - 1e-9)
        res = least_squares(residuals, x0, args=(pairs, idx), loss="soft_l1", f_scale=1.5, max_nfev=200, bounds=(lo, hi))
        x0 = res.x
        k1, k2, f, px, py = x0[:5]
        print(f"tol {tol}: k1 {k1:+.5f} k2 {k2:+.5f} f {f:.1f} pole ({px:.1f},{py:.1f}) cost {res.cost:.1f}", flush=True)
        # re-pair with the fitted model at a tighter tolerance
        dref = D(ref[:, :2], k1, k2); tree2 = cKDTree(dref)
        newpairs = []
        for j, i in enumerate(idx):
            cur = stars[i]
            pred = R.apply(Hmat(x0[5 + j], f, (px, py)), D(cur[:, :2], k1, k2))
            d, jj = tree2.query(pred, distance_upper_bound=tol)
            ok = np.isfinite(d)
            newpairs.append((ref[jj[ok], :2], cur[ok, :2]))
        pairs = newpairs
    r = residuals(x0, pairs, idx).reshape(-1, 2)
    e = np.hypot(r[:, 0], r[:, 1])
    allref = np.concatenate([a for a, _ in pairs])
    rad = np.hypot(allref[:, 0] - R.CX, allref[:, 1] - R.CY)
    for lo, hi in ((0, 1500), (1500, 2500), (2500, 3500), (3500, 5000)):
        m = (rad >= lo) & (rad < hi)
        print(f"  r {lo}-{hi}: n {m.sum()} median resid {np.median(e[m]):.2f} px")
    H = np.zeros((N + 1, 3, 3)); H[R.REF] = np.eye(3)
    for j, i in enumerate(idx):
        H[i] = Hmat(x0[5 + j], f, (px, py))
    have = np.zeros(N + 1, bool); have[R.REF] = True; have[idx] = True
    np.savez(f"{W}/model.npz", k1=k1, k2=k2, f=f, pole=np.array([px, py]), H=H, have=have)
    print("saved model; frames with H:", have.sum())


if __name__ == "__main__":
    main()
