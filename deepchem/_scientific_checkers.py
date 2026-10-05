"""Opt-in scientific runtime checks used by the SciBench pilot.

The module is inert unless ``SCIBENCH_TRIGGER_LOG`` names a log file. Checks
append stable IDs and never raise into DeepChem, never change a return value
and restore any global random state they touch.

Tolerance convention (SANITIZER.md 5.8): ``_C * eps(dtype) * size * scale``
with each factor justified at the call site.
"""

from __future__ import annotations

import json
import math
import os
from contextlib import contextmanager

import numpy as np

_ACTIVE = False
_EPS = float(np.finfo(np.float64).eps)
_C = 64.0
_HUGE = 1e100


def enabled():
    return bool(os.environ.get("SCIBENCH_TRIGGER_LOG"))


def trigger(checker_id):
    path = os.environ.get("SCIBENCH_TRIGGER_LOG")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"checker_id": checker_id}) + "\n")
    except Exception:
        _swallowed()


def trigger_if(condition, checker_id):
    _reach(checker_id)
    if bool(condition):
        trigger(checker_id)


SWALLOWED = []
REACHED = {}


def _reach(checker_id):
    """Curator-only: count predicate evaluations (observation reachability)."""
    if os.environ.get("SCIBENCH_CHECKER_DEBUG"):
        REACHED[checker_id] = REACHED.get(checker_id, 0) + 1


def _swallowed():
    """Curator-only: record exceptions swallowed inside a checker."""
    if os.environ.get("SCIBENCH_CHECKER_DEBUG"):
        import traceback
        SWALLOWED.append(traceback.format_exc())


@contextmanager
def _checking():
    """Re-entrancy guard: probes that call the public API do not re-check."""
    global _ACTIVE
    if _ACTIVE:
        yield False
        return
    _ACTIVE = True
    try:
        yield True
    finally:
        _ACTIVE = False


# ----------------------------------------------------------------- helpers
def _np(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _f64(x):
    a = _np(x)
    return a.astype(np.complex128 if a.dtype.kind == "c" else np.float64)


def _eps_of(x):
    """Machine epsilon of the (real part of the) dtype of a numpy/torch value."""
    try:
        if hasattr(x, "dtype") and not isinstance(x, np.ndarray):
            import torch
            if isinstance(x, torch.Tensor):
                return float(torch.finfo(x.dtype).eps)
        a = np.asarray(x)
        if a.dtype.kind in "fc":
            return float(np.finfo(a.dtype).eps)
    except Exception:
        _swallowed()
    return _EPS


def _finite(*values):
    try:
        return all(np.all(np.isfinite(_np(v))) for v in values)
    except Exception:
        return False


def _sane(*values, hi=_HUGE):
    """Finite and below ``hi`` in magnitude (guards overflow in squares)."""
    try:
        for v in values:
            a = _np(v)
            if a.size and (not np.all(np.isfinite(a)) or np.max(np.abs(a)) > hi):
                return False
        return True
    except Exception:
        return False


def _maxabs(*values):
    out = 0.0
    for v in values:
        a = _np(v)
        if a.size:
            out = max(out, float(np.max(np.abs(a))))
    return out


def _violates(diff, tol):
    """True when ``diff`` is not provably within ``tol`` (NaN alarms)."""
    return not (diff <= tol)


_R0 = None


def _rotation():
    """A fixed generic proper rotation (axis (1,2,3), angle 0.9 rad)."""
    global _R0
    if _R0 is None:
        axis = np.array([1.0, 2.0, 3.0])
        axis /= np.linalg.norm(axis)
        K = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]],
                      [-axis[1], axis[0], 0.0]])
        th = 0.9
        _R0 = np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)
    return _R0


def _rigid(points, scale):
    """Rotate then translate; translation scales with the cloud (8.2)."""
    t = scale * np.array([0.37, -0.61, 0.83])
    return np.asarray(points, float) @ _rotation().T + t, t


# ================================================================ A. SO(3)
def _wigner_ok(k, alpha, beta, gamma):
    import torch
    k = int(k)
    if k < 0 or k > 32:
        return None
    angles = [torch.as_tensor(a).detach() for a in (alpha, beta, gamma)]
    if not all(bool(torch.isfinite(a).all()) for a in angles):
        return None
    mx = max(float(a.abs().max()) if a.numel() else 0.0 for a in angles)
    if mx > 1e6:
        return None
    return k, mx


def check_wigner_d(k, alpha, beta, gamma, D):
    """DC-EQ-001 (rotation matrix) and DC-EQ-002 (inverse law)."""
    try:
        with _checking() as run:
            if not run:
                return
            import torch
            ok = _wigner_ok(k, alpha, beta, gamma)
            if ok is None or not isinstance(D, torch.Tensor):
                return
            k, mx = ok
            if D.is_complex() or not bool(torch.isfinite(D).all()):
                return
            n = 2 * k + 1
            # The generators are built in torch's default dtype regardless of
            # the angle dtype, so exp(2*pi*G) = I only to that precision and
            # the 2*pi range reduction inherits it: pin eps to the coarser of
            # the two (SANITIZER.md 5.8 T, "dtype actually used").
            eps = max(_eps_of(D), float(torch.finfo(torch.get_default_dtype()).eps))
            # matrix_exp error ~ eps*||A||, ||A|| <= 2*pi*k after the mod; the
            # mod itself loses eps*|angle| of angle, amplified by k.
            amp = 1.0 + 2 * math.pi * k + k * mx
            tol = _C * eps * n * amp
            Dd = D.detach().to(torch.float64)
            eye = torch.eye(n, dtype=torch.float64)
            orth = float((Dd @ Dd.transpose(-1, -2) - eye).abs().max())
            det = float((torch.linalg.det(Dd) - 1).abs().max())
            trigger_if(_violates(orth, tol) or _violates(det, tol * n),
                       "DC-EQ-001")
            from deepchem.utils.equivariance_utils import wigner_D
            a, b, g = (torch.as_tensor(x) for x in (alpha, beta, gamma))
            D2 = wigner_D(k, -g, -b, -a)
            prod = Dd @ D2.detach().to(torch.float64)
            resid = float((prod - eye).abs().max())
            tol2 = 2 * tol
            trigger_if(_violates(resid, tol2), "DC-EQ-002")
    except Exception:
        _swallowed()


def check_so3_generators(k, G):
    """DC-EQ-003: skew-symmetry and degree-independent structure constants."""
    try:
        with _checking() as run:
            if not run:
                return
            import torch
            k = int(k)
            if k < 1 or k > 32 or not _finite(G):
                return
            from deepchem.utils.equivariance_utils import so3_generators
            ref = so3_generators(1)
            Gd = G.detach().to(torch.float64)
            Rd = ref.detach().to(torch.float64)
            eps = max(_eps_of(G), _eps_of(ref))
            tol = _C * eps * (k + 1)**2
            skew = float((Gd + Gd.transpose(-1, -2)).abs().max())
            worst = skew

            def comm(X, a, b):
                return X[a] @ X[b] - X[b] @ X[a]

            for a in range(3):
                for b in range(3):
                    C1 = comm(Rd, a, b)
                    pred = torch.zeros_like(Gd[0])
                    for c in range(3):
                        denom = float((Rd[c] * Rd[c]).sum())
                        f = float((Rd[c] * C1).sum()) / denom
                        pred = pred + f * Gd[c]
                    worst = max(worst, float((comm(Gd, a, b) - pred).abs().max()))
            trigger_if(_violates(worst, tol), "DC-EQ-003")
    except Exception:
        _swallowed()


def check_su2_generators(k, G):
    """DC-EQ-004: quadratic Casimir sum_a G_a^2 = -k(k+1) I."""
    try:
        with _checking() as run:
            if not run:
                return
            import torch
            k = int(k)
            if k < 0 or k > 64 or not _finite(G):
                return
            Gd = G.detach().to(torch.complex128)
            n = 2 * k + 1
            cas = (Gd @ Gd).sum(dim=0)
            resid = float((cas + k * (k + 1) * torch.eye(n, dtype=torch.complex128)
                           ).abs().max())
            tol = _C * _eps_of(G) * (k + 1)**3 + _C * _EPS
            trigger_if(_violates(resid, tol), "DC-EQ-004")
    except Exception:
        _swallowed()


def check_spherical_harmonics(d, theta, phi, Y):
    """DC-EQ-005: sum_m Y_lm^2 = (2l+1)/(4 pi) in every direction."""
    try:
        with _checking() as run:
            if not run:
                return
            import torch
            d = int(d)
            if d < 0 or d > 40 or not _sane(theta, phi, hi=1e8):
                return
            Yd = Y.detach().to(torch.float64)
            shell = (Yd * Yd).sum(dim=-1)
            target = (2 * d + 1) / (4 * math.pi)
            diff = float((shell - target).abs().max())
            tol = 16 * _eps_of(Y) * (d + 1)**2 * target
            trigger_if(_violates(diff, tol), "DC-EQ-005")
    except Exception:
        _swallowed()


def check_so3_exp(w, R):
    """DC-EQ-007: exp of an axis-angle vector is a proper rotation."""
    try:
        with _checking() as run:
            if not run:
                return
            import torch
            if not _sane(w, hi=1e6) or not _finite(R):
                return
            Rd = R.detach().to(torch.float64)
            eye = torch.eye(3, dtype=torch.float64)
            orth = float((Rd @ Rd.transpose(-1, -2) - eye).abs().max())
            det = float((torch.linalg.det(Rd) - 1).abs().max())
            tol = _C * _eps_of(R) * 3
            trigger_if(_violates(orth, tol) or _violates(det, 3 * tol),
                       "DC-EQ-007")
    except Exception:
        _swallowed()


def check_so3_log(R, w):
    """DC-EQ-008: exp(log(R)) recovers R away from the cut locus."""
    try:
        with _checking() as run:
            if not run:
                return
            import torch
            if not _finite(R, w):
                return
            Rd = R.detach().to(torch.float64)
            eye = torch.eye(3, dtype=torch.float64)
            orth = float((Rd @ Rd.transpose(-1, -2) - eye).abs().max())
            if orth > 1e-3 or float(torch.linalg.det(Rd).min()) < 0.5:
                return
            tr = Rd[..., 0, 0] + Rd[..., 1, 1] + Rd[..., 2, 2]
            theta = torch.acos(((tr - 1) / 2).clamp(-1, 1))
            gap = float(math.pi - float(theta.max()))
            if gap < 5e-2:
                return
            from deepchem.utils.equivariance_utils import SO3
            back = SO3().exp(w.detach().to(R.dtype))
            diff = float((back.detach().to(torch.float64) - Rd).abs().max())
            # acos(cos) and theta/sin(theta) each amplify by 1/sin(theta):
            # the conditioning of log near the cut locus is ~1/gap^2.
            tol = _C * _eps_of(R) * (1 + math.pi / gap**2) + 4 * orth
            trigger_if(_violates(diff, tol), "DC-EQ-008")
    except Exception:
        _swallowed()


def check_spherical_chart(cartesian, divide_radius_by, spherical):
    """DC-EQ-010: radial = |v|/c and angles are invariant under v -> 2v."""
    try:
        with _checking() as run:
            if not run:
                return
            import torch
            if not _sane(cartesian, hi=1e15) or not _finite(spherical):
                return
            c = float(divide_radius_by)
            if not (c > 0 and math.isfinite(c)):
                return
            v = cartesian.detach().to(torch.float64)
            nrm = torch.linalg.norm(v, dim=-1)
            fi = torch.finfo(cartesian.dtype)
            # squares and their sum must be representable in the working dtype
            if (float(nrm.min()) < 1e3 * math.sqrt(fi.tiny) or
                    float(nrm.max()) > math.sqrt(fi.max) / 1e3):
                return
            eps = _eps_of(spherical)
            expected = nrm / c
            rad = spherical[..., 0].detach().to(torch.float64)
            r_err = float(((rad - expected).abs() / expected).max())
            from deepchem.utils.equivariance_utils import \
                get_spherical_from_cartesian
            doubled = get_spherical_from_cartesian(2 * cartesian, divide_radius_by)
            ang = float((doubled[..., 1:].detach().to(torch.float64) -
                         spherical[..., 1:].detach().to(torch.float64)).abs().max())
            rad2 = float(((doubled[..., 0].detach().to(torch.float64) -
                           2 * rad).abs() / (2 * expected)).max())
            trigger_if(_violates(r_err, 16 * eps) or _violates(ang, 16 * eps) or
                       _violates(rad2, 16 * eps), "DC-EQ-010")
    except Exception:
        _swallowed()


def check_basis_change(k, Q):
    """DC-EQ-011: the real-to-complex harmonic basis change is unitary."""
    try:
        with _checking() as run:
            if not run:
                return
            import torch
            k = int(k)
            if k < 0 or k > 64 or not _finite(Q):
                return
            Qd = Q.detach().to(torch.complex128)
            n = 2 * k + 1
            resid = float((Qd.conj().T @ Qd -
                           torch.eye(n, dtype=torch.complex128)).abs().max())
            tol = _C * _eps_of(Q) * n
            trigger_if(_violates(resid, tol), "DC-EQ-011")
    except Exception:
        _swallowed()


_CG_SEEN = set()


def check_cg_intertwiner(J, d_in, d_out, Q):
    """DC-EQ-012: Q_J intertwines (D_out x D_in) with D_J at fresh angles."""
    try:
        with _checking() as run:
            if not run:
                return
            import torch
            J, d_in, d_out = int(J), int(d_in), int(d_out)
            if max(J, d_in, d_out) > 4 or not _finite(Q):
                return
            Qd = Q.detach().to(torch.float64)
            key = (J, d_in, d_out, Qd.numpy().tobytes())
            if key in _CG_SEEN:
                return
            _CG_SEEN.add(key)
            m_in, m_out, m = 2 * d_in + 1, 2 * d_out + 1, 2 * J + 1
            if tuple(Qd.shape) != (m_in * m_out, m):
                trigger("DC-EQ-012")
                return
            from deepchem.utils.equivariance_utils import irr_repr, kron
            old = torch.get_default_dtype()
            torch.set_default_dtype(torch.float64)
            try:
                worst = 0.0
                for ang in ((0.7, 1.9, 2.3), (3.1, 0.4, 5.2)):
                    Dt = kron(irr_repr(d_out, *ang), irr_repr(d_in, *ang))
                    DJ = irr_repr(J, *ang)
                    worst = max(worst, float((Dt @ Qd - Qd @ DJ).abs().max()))
            finally:
                torch.set_default_dtype(old)
            gram = Qd.T @ Qd
            c = float(torch.trace(gram)) / m
            schur = float((gram - c * torch.eye(m, dtype=torch.float64)).abs().max())
            # the basis is a kernel vector with singular values < 1e-10
            # (library constant); worst-case residual scales with that.
            tol = 1e-6 * max(m_in * m_out, 1)
            trigger_if(_violates(worst, tol) or _violates(schur, tol),
                       "DC-EQ-012")
    except Exception:
        _swallowed()


# ============================================================ B. VMC sampler
def check_electron_init(sampler, no_sample, stddev):
    """DC-ES-001: electron-count conservation and block placement."""
    try:
        with _checking() as run:
            if not run:
                return
            cv = np.asarray(sampler.central_value, float)
            x = np.asarray(sampler.x)
            if cv.ndim != 2 or x.ndim != 4 or not _sane(cv, x, stddev):
                return
            counts = [int(np.asarray(no_sample[i]).reshape(-1)[0])
                      for i in range(cv.shape[0])]
            if min(counts) < 0 or float(stddev) < 0:
                return
            b, ndim = int(sampler.batch_no), cv.shape[1]
            _reach("DC-ES-001")
            if x.shape != (b, sum(counts), 1, ndim):
                trigger("DC-ES-001")
                return
            start = 0
            for k, n_k in enumerate(counts):
                if n_k == 0:
                    continue
                block = x[:, start:start + n_k]
                start += n_k
                mean = block.mean(axis=(0, 1, 2))
                tol = (6.0 * float(stddev) / math.sqrt(b * n_k) +
                       _C * _EPS * (1 + _maxabs(cv[k])))
                if np.any(np.abs(mean - cv[k]) > tol):
                    trigger("DC-ES-001")
                    return
    except Exception:
        _swallowed()


def check_harmonic_mean(sampler, y, result):
    """DC-ES-002: harmonic-mean distance lies in [min d, max d]."""
    try:
        with _checking() as run:
            if not run:
                return
            y = np.asarray(y, float)
            cv = np.asarray(sampler.central_value, float)
            if not _sane(y, cv, result):
                return
            d = np.linalg.norm(y - cv, axis=-1, keepdims=True)
            if not np.all(d > 0):
                return
            dmin = d.min(axis=-2, keepdims=True)
            dmax = d.max(axis=-2, keepdims=True)
            h = np.asarray(result, float)
            slack = _C * _EPS * cv.shape[0]
            bad = np.any(h < dmin * (1 - slack)) or np.any(h > dmax * (1 + slack))
            trigger_if(bad or not np.all(np.isfinite(h)), "DC-ES-002")
    except Exception:
        _swallowed()


def check_flat_target_acceptance(sampler, stddev):
    """DC-ES-003: constant target + symmetric proposal accepts every move."""
    try:
        with _checking() as run:
            if not run:
                return
            if not (sampler.simultaneous and sampler.symmetric):
                return
            x = np.asarray(sampler.x)
            if x.ndim != 4 or x.size == 0 or not _sane(x, stddev, hi=1e12):
                return
            if float(stddev) < 0 or not (1 <= int(sampler.steps) <= 200):
                return
            from deepchem.utils.electron_sampler import ElectronSampler
            state = np.random.get_state()
            try:
                probe = ElectronSampler(
                    central_value=sampler.central_value,
                    f=lambda z: np.zeros(np.shape(z)[0]),
                    batch_no=sampler.batch_no,
                    x=np.copy(x),
                    steps=int(sampler.steps),
                    steps_per_update=sampler.steps_per_update,
                    seed=None,
                    symmetric=True,
                    simultaneous=True)
                acc = probe.move(stddev=float(stddev))
            finally:
                np.random.set_state(state)
            trigger_if(_violates(abs(float(acc) - 1.0), 1e-12), "DC-ES-003")
    except Exception:
        _swallowed()


def check_log_prob_gaussian(sampler, y, mu, sigma, result):
    """DC-ES-004: Gaussian dependence of the log-density on sigma and y."""
    try:
        with _checking() as run:
            if not run:
                return
            y = np.asarray(y, float)
            mu = np.asarray(mu, float)
            sigma = np.asarray(sigma, float)
            if y.ndim != 4 or not _sane(y, mu, sigma, hi=1e50):
                return
            if np.min(sigma) <= 0 or np.max(np.abs(np.log(sigma))) > 50:
                return
            sig = np.broadcast_to(sigma, y.shape)
            n_coords = float(np.prod(y.shape[1:]))
            f = sampler.log_prob_gaussian
            base = np.asarray(f(mu, mu, sigma), float)
            doubled = np.asarray(f(mu, mu, 2.0 * sigma), float)
            scale_term = n_coords * math.log(2.0)
            err1 = np.max(np.abs((doubled - base) + scale_term))
            quad = -0.5 * np.sum(((y - mu) / sig)**2, axis=(1, 2, 3))
            err2 = np.max(np.abs((np.asarray(result, float) - base) - quad))
            mag = (n_coords * (1 + np.max(np.abs(np.log(sig)))) +
                   np.max(np.abs(quad)))
            tol = _C * _EPS * mag
            trigger_if(_violates(err1, tol) or _violates(err2, tol),
                       "DC-ES-004")
    except Exception:
        _swallowed()


# ============================================================ C. geometry
def check_random_rotation(R):
    """DC-GU-001: a generated rotation matrix is proper orthogonal."""
    try:
        with _checking() as run:
            if not run:
                return
            R = np.asarray(R, float)
            if R.shape != (3, 3) or not _finite(R):
                return
            tol = _C * _EPS * 3
            orth = np.max(np.abs(R @ R.T - np.eye(3)))
            trigger_if(_violates(orth, tol) or
                       _violates(abs(np.linalg.det(R) - 1), tol), "DC-GU-001")
    except Exception:
        _swallowed()


def check_rotate_molecules(before, after):
    """DC-GU-002: rotate_molecules preserves distances and origin radii."""
    try:
        with _checking() as run:
            if not run:
                return
            if len(before) != len(after):
                trigger("DC-GU-002")
                return
            for a, b in zip(before, after):
                a, b = np.asarray(a, float), np.asarray(b, float)
                if a.ndim != 2 or a.shape[1] != 3 or a.shape[0] > 300:
                    continue
                if not _sane(a, hi=1e100) or not _finite(b):
                    continue
                _reach("DC-GU-002")
                if a.shape != b.shape:
                    trigger("DC-GU-002")
                    return
                tol = _C * _EPS * max(_maxabs(a), 1e-300) * 3
                da = np.linalg.norm(a[:, None] - a[None], axis=-1)
                db = np.linalg.norm(b[:, None] - b[None], axis=-1)
                ra = np.linalg.norm(a, axis=-1)
                rb = np.linalg.norm(b, axis=-1)
                if (_violates(np.max(np.abs(da - db)), tol) or
                        _violates(np.max(np.abs(ra - rb)), tol)):
                    trigger("DC-GU-002")
                    return
    except Exception:
        _swallowed()


def check_angle_between(vi, vj, angle):
    """DC-GU-003: atan2 form, argument symmetry, scale invariance."""
    try:
        with _checking() as run:
            if not run:
                return
            a, b = np.asarray(vi, float), np.asarray(vj, float)
            if a.ndim != 1 or a.shape != b.shape or not _sane(a, b, hi=1e100):
                return
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if not (na > 1e-100 and nb > 1e-100):
                return
            u, v = a / na, b / nb
            ref = math.atan2(float(np.linalg.norm(np.cross(
                np.pad(u, (0, 3 - u.size)), np.pad(v, (0, 3 - v.size))))),
                             float(u @ v))
            # arccos conditioning ~ 1/sin(theta); below sqrt(eps) it cannot
            # resolve the angle at all.
            tol = _C * _EPS / max(math.sin(ref), 1e-8)
            from deepchem.utils.geometry_utils import angle_between
            sw = angle_between(vj, vi)
            sc = angle_between(2.0 * a, 4.0 * b)
            bad = (_violates(abs(float(angle) - ref), tol) or
                   _violates(abs(float(sw) - float(angle)), tol) or
                   _violates(abs(float(sc) - float(angle)), tol))
            trigger_if(bad, "DC-GU-003")
    except Exception:
        _swallowed()


def check_pairwise_distances(A, B, D):
    """DC-GU-004: rigid-motion invariance and transpose symmetry."""
    try:
        with _checking() as run:
            if not run:
                return
            A, B, D = np.asarray(A, float), np.asarray(B, float), np.asarray(D)
            if (A.ndim != 2 or B.ndim != 2 or A.shape[1] != 3 or B.shape[1] != 3
                    or not _sane(A, B, hi=1e100)):
                return
            scale = max(_maxabs(A, B), 1e-300)
            from deepchem.utils.geometry_utils import compute_pairwise_distances
            A2, t = _rigid(A, scale)
            B2 = B @ _rotation().T + t
            D2 = compute_pairwise_distances(A2, B2)
            tol = _C * _EPS * (scale + np.linalg.norm(t)) * 4
            sym = compute_pairwise_distances(B, A)
            bad = (D2.shape != D.shape or
                   _violates(np.max(np.abs(D2 - D)), tol) or
                   _violates(np.max(np.abs(sym.T - D)),
                             8 * _EPS * max(_maxabs(D), 1e-300)))
            trigger_if(bad, "DC-GU-004")
    except Exception:
        _swallowed()


def check_angle_cutoff(vi, vj, cutoff, result):
    """DC-GU-005: predicate is symmetric in the vectors, monotone in cutoff."""
    try:
        with _checking() as run:
            if not run:
                return
            a, b = np.asarray(vi, float), np.asarray(vj, float)
            c = float(cutoff)
            if (a.ndim != 1 or a.shape != b.shape or not _sane(a, b, hi=1e100) or
                    not (0 < c <= 180) or np.linalg.norm(a) < 1e-100 or
                    np.linalg.norm(b) < 1e-100):
                return
            from deepchem.utils.geometry_utils import is_angle_within_cutoff
            r = bool(result)
            u, v = a / np.linalg.norm(a), b / np.linalg.norm(b)
            pad = lambda x: np.pad(x, (0, 3 - x.size))
            ref = math.degrees(math.atan2(
                float(np.linalg.norm(np.cross(pad(u), pad(v)))), float(u @ v)))
            # a 1-ulp difference in the dot product may flip a predicate that
            # sits exactly on its threshold: only test symmetry away from it
            on_edge = abs(ref - (180.0 - c)) <= 1e-9 * 180.0
            bad = (not on_edge) and bool(is_angle_within_cutoff(vj, vi, c)) != r
            if r:
                bad = bad or not is_angle_within_cutoff(vi, vj, min(180.0, 1.5 * c))
            else:
                bad = bad or bool(is_angle_within_cutoff(vi, vj, 0.5 * c))
            trigger_if(bad, "DC-GU-005")
    except Exception:
        _swallowed()


# ============================================================== D. boxes
def _ranges(box):
    return (tuple(box.x_range), tuple(box.y_range), tuple(box.z_range))


def _inside(inner, outer):
    return all(o[0] <= i[0] and i[1] <= o[1]
               for i, o in zip(_ranges(inner), _ranges(outer)))


def check_box_intersection(box1, box2, result):
    """DC-BX-001: a non-empty intersection lies in both boxes."""
    try:
        with _checking() as run:
            if not run:
                return
            r1, r2 = _ranges(box1), _ranges(box2)
            if not all(max(a[0], b[0]) <= min(a[1], b[1])
                       for a, b in zip(r1, r2)):
                return
            bad = not (_inside(result, box1) and _inside(result, box2))
            trigger_if(bad, "DC-BX-001")
    except Exception:
        _swallowed()


def check_box_merge(boxes, merged):
    """DC-BX-003: merging never discards an input box or leaves the hull."""
    try:
        with _checking() as run:
            if not run:
                return
            if len(boxes) == 0:
                return
            _reach("DC-BX-003")
            lo = [min(r[ax][0] for r in map(_ranges, boxes)) for ax in range(3)]
            hi = [max(r[ax][1] for r in map(_ranges, boxes)) for ax in range(3)]
            for b in boxes:
                if not any(_inside(b, m) for m in merged):
                    trigger("DC-BX-003")
                    return
            for m in merged:
                rr = _ranges(m)
                if any(rr[ax][0] < lo[ax] or rr[ax][1] > hi[ax] for ax in range(3)):
                    trigger("DC-BX-003")
                    return
    except Exception:
        _swallowed()


# ===================================================== E. non-covalent geometry
def _pairs(contacts):
    return {(int(a), int(b)) for a, b in contacts}


def _hbond_margins_ok(frag1, frag2, D, bins, cutoffs):
    """No quantity used by the predicate is within rounding of a threshold."""
    x1, x2 = np.asarray(frag1[0], float), np.asarray(frag2[0], float)
    scale = max(_maxabs(x1, x2), 1.0)
    if not _sane(x1, x2, D, hi=1e100):
        return False
    from scipy.spatial.distance import cdist
    ref = cdist(x1, x2)
    if D.shape != ref.shape or np.max(np.abs(D - ref)) > 1e-9 * scale:
        return False
    m = 1e-9 * scale
    for lo, hi in bins:
        if np.any(np.abs(D - lo) <= m) or np.any(np.abs(D - hi) <= m):
            return False
    z1 = np.array([a.GetAtomicNum() for a in frag1[1].GetAtoms()])
    z2 = np.array([a.GetAtomicNum() for a in frag2[1].GetAtoms()])
    no1, no2 = np.where((z1 == 7) | (z1 == 8))[0], np.where((z2 == 7) | (z2 == 8))[0]
    h = [(x1[z1 == 1], x1[no1]), (x2[z2 == 1], x2[no2])]
    for hs, heavy in h:
        if hs.size and heavy.size and np.any(np.abs(cdist(hs, heavy) - 1.3) <= m):
            return False
    hs_all = np.vstack([x1[z1 == 1], x2[z2 == 1]])
    if hs_all.size and no1.size and no2.size:
        if hs_all.shape[0] * no1.size * no2.size > 2_000_000:
            return False
        v1 = x1[no1][None, :, None, :] - hs_all[:, None, None, :]
        v2 = x2[no2][None, None, :, :] - hs_all[:, None, None, :]
        n1, n2 = np.linalg.norm(v1, axis=-1), np.linalg.norm(v2, axis=-1)
        if np.any(n1 == 0) or np.any(n2 == 0):
            return False
        ang = np.degrees(np.arctan2(np.linalg.norm(np.cross(v1, v2), axis=-1),
                                    np.sum(v1 * v2, axis=-1)))
        # rigid-motion rounding eps*scale perturbs an angle by eps*scale/|v|
        margin = np.degrees(1e3 * _EPS * scale / np.minimum(n1, n2)) + 1e-6
        for c in cutoffs:
            if np.any(np.abs(ang - (180.0 - c)) <= margin):
                return False
    return True


def check_hydrogen_bonds(frag1, frag2, D, bins, cutoffs, result):
    """DC-NC-002 (rigid motion) and DC-NC-003 (fragment exchange)."""
    try:
        with _checking() as run:
            if not run:
                return
            D = np.asarray(D, float)
            if not _hbond_margins_ok(frag1, frag2, D, bins, cutoffs):
                return
            from deepchem.utils.noncovalent_utils import compute_hydrogen_bonds
            x1, x2 = np.asarray(frag1[0], float), np.asarray(frag2[0], float)
            scale = max(_maxabs(x1, x2), 1.0)
            y1, t = _rigid(x1, scale)
            y2 = x2 @ _rotation().T + t
            from scipy.spatial.distance import cdist
            moved = compute_hydrogen_bonds((y1, frag1[1]), (y2, frag2[1]),
                                           cdist(y1, y2), bins, cutoffs)
            orig = [_pairs(c) for c in result]
            trigger_if([_pairs(c) for c in moved] != orig, "DC-NC-002")
            swapped = compute_hydrogen_bonds(frag2, frag1, D.T, bins, cutoffs)
            sw = [{(j, i) for i, j in _pairs(c)} for c in swapped]
            trigger_if(sw != orig, "DC-NC-003")
    except Exception:
        _swallowed()


def check_salt_bridges(first, second, contacts):
    """DC-NC-004: salt-bridge partners carry opposite-sign charge."""
    try:
        with _checking() as run:
            if not run:
                return
            from deepchem.utils.fragment_utils import get_partial_charge
            fa, sa = first.GetAtoms(), second.GetAtoms()
            for i, j in contacts:
                qi = get_partial_charge(fa[int(i)])
                qj = get_partial_charge(sa[int(j)])
                if not (math.isfinite(qi) and math.isfinite(qj)):
                    continue
                _reach("DC-NC-004")
                if not qi * qj < 0:
                    trigger("DC-NC-004")
                    return
    except Exception:
        _swallowed()


def check_cation_pi(cation, center, normal, dist_cutoff, angle_cutoff, result):
    """DC-NC-005: rigid-motion and normal-sign invariance."""
    try:
        with _checking() as run:
            if not run:
                return
            c, o, n = (np.asarray(v, float) for v in (cation, center, normal))
            if not (c.shape == o.shape == n.shape == (3,)):
                return
            if not _sane(c, o, n, hi=1e100):
                return
            dc, ac = float(dist_cutoff), float(angle_cutoff)
            if not (dc > 0 and 0 < ac < 90):
                return
            vec = c - o
            dist = np.linalg.norm(vec)
            if dist < 1e-100 or np.linalg.norm(n) < 1e-100:
                return
            scale = max(_maxabs(c, o), 1.0)
            ang = math.degrees(math.atan2(np.linalg.norm(np.cross(vec, n)),
                                          float(vec @ n)))
            # no tested quantity within rounding of its threshold; the angle
            # of (cation - center) carries eps*scale/|vec| of rigid-motion error
            amargin = math.degrees(1e3 * _EPS * scale / dist) + 1e-7
            if (abs(dist - dc) <= 1e-9 * scale or abs(ang - ac) <= amargin or
                    abs(ang - (180 - ac)) <= amargin):
                return
            from deepchem.utils.noncovalent_utils import is_cation_pi
            c2, t = _rigid(c, scale)
            o2 = o @ _rotation().T + t
            n2 = n @ _rotation().T
            moved = is_cation_pi(c2, o2, n2, dist_cutoff, angle_cutoff)
            flipped = is_cation_pi(c, o, -n, dist_cutoff, angle_cutoff)
            trigger_if(bool(moved) != bool(result) or
                       bool(flipped) != bool(result), "DC-NC-005")
    except Exception:
        _swallowed()


def check_ring_normal(mol, ring_indices, normal):
    """DC-NC-001: the returned normal is the ring-plane normal (planar rings)."""
    try:
        with _checking() as run:
            if not run:
                return
            idx = list(ring_indices)
            if len(idx) < 3:
                return
            conf = mol.GetConformer()
            pts = np.array([list(conf.GetAtomPosition(int(i))) for i in idx])
            nrm = np.asarray(normal, float)
            if not _sane(pts, nrm, hi=1e100) or np.linalg.norm(nrm) < 1e-100:
                return
            centered = pts - pts.mean(axis=0)
            _, _, vt = np.linalg.svd(centered)
            n_svd = vt[-1]
            dev = float(np.max(np.abs(centered @ n_svd)))
            diam = float(np.max(np.linalg.norm(pts[:, None] - pts[None], axis=-1)))
            if diam <= 0 or dev > 1e-6 * diam:
                return
            v1, v2 = pts[1] - pts[0], pts[2] - pts[0]
            l1, l2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if l1 <= 0 or l2 <= 0:
                return
            s0 = np.linalg.norm(np.cross(v1, v2)) / (l1 * l2)
            if s0 < 1e-3:
                return
            sin_ang = float(np.linalg.norm(np.cross(nrm / np.linalg.norm(nrm),
                                                    n_svd)))
            tol = _C * _EPS / s0 + 4 * dev / (min(l1, l2) * s0)
            trigger_if(_violates(sin_ang, tol), "DC-NC-001")
    except Exception:
        _swallowed()


# =========================================================== F. fragments
def _rigid_fragments(fragments):
    allx = [np.asarray(f[0], float) for f in fragments]
    scale = max(_maxabs(*allx), 1.0)
    t = scale * np.array([0.37, -0.61, 0.83])
    return [(x @ _rotation().T + t, f[1]) for x, f in zip(allx, fragments)], scale


def check_contact_atoms(fragments, cutoff, result):
    """DC-FR-001: contact selection is rigid-motion and order invariant."""
    try:
        with _checking() as run:
            if not run:
                return
            from scipy.spatial.distance import cdist
            xs = [np.asarray(f[0], float) for f in fragments]
            if len(xs) < 2 or sum(len(x) for x in xs) > 3000:
                return
            if not _sane(*xs, hi=1e100) or not np.isfinite(float(cutoff)):
                return
            scale = max(_maxabs(*xs), 1.0)
            for a in range(len(xs)):
                for b in range(a + 1, len(xs)):
                    if np.any(np.abs(cdist(xs[a], xs[b]) - cutoff) <= 1e-9 * scale):
                        return
            from deepchem.utils.fragment_utils import get_contact_atom_indices
            moved, _ = _rigid_fragments(fragments)
            m = get_contact_atom_indices(moved, cutoff)
            rev = get_contact_atom_indices(list(reversed(fragments)), cutoff)
            orig = [list(map(int, r)) for r in result]
            trigger_if([list(map(int, r)) for r in m] != orig or
                       [list(map(int, r)) for r in reversed(rev)] != orig,
                       "DC-FR-001")
    except Exception:
        _swallowed()


def check_strip_hydrogens(coords, mol, result):
    """DC-FR-002: stripping conserves exactly the heavy atoms and coordinates."""
    try:
        with _checking() as run:
            if not run:
                return
            coords = np.asarray(coords)
            z = [a.GetAtomicNum() for a in mol.GetAtoms()]
            if coords.shape != (len(z), 3):
                return
            out_coords, frag = result
            keep = [i for i, zi in enumerate(z) if zi != 1]
            out_z = [a.GetAtomicNum() for a in frag.GetAtoms()]
            bad = (out_z != [z[i] for i in keep] or 1 in out_z or
                   not np.array_equal(np.asarray(out_coords), coords[keep]) or
                   not np.array_equal(np.asarray(frag.GetCoords()), coords[keep]))
            trigger_if(bad, "DC-FR-002")
    except Exception:
        _swallowed()


def check_merge_fragments(fragments, merged):
    """DC-FR-003: merging conserves atoms, charges and coordinates."""
    try:
        with _checking() as run:
            if not run:
                return
            if len(fragments) == 0 or merged is None:
                return
            def rows(f):
                return [(a.GetAtomicNum(), float(a.GetPartialCharge()))
                        for a in f.GetAtoms()]
            expect = [r for f in fragments for r in rows(f)]
            xs = np.concatenate([np.asarray(f.GetCoords()) for f in fragments])
            bad = (rows(merged) != expect or
                   not np.array_equal(np.asarray(merged.GetCoords()), xs))
            trigger_if(bad, "DC-FR-003")
    except Exception:
        _swallowed()


# =========================================================== G. Coulomb matrix
_BOHR_ANGSTROM = None


def _bohr():
    global _BOHR_ANGSTROM
    if _BOHR_ANGSTROM is None:
        from scipy import constants
        _BOHR_ANGSTROM = constants.physical_constants["Bohr radius"][0] * 1e10
    return _BOHR_ANGSTROM


def _conformer_arrays(mol):
    return [np.array(c.GetPositions(), float) for c in mol.GetConformers()]


def check_coulomb_matrix(featurizer, mol, result):
    """DC-CM-001 (rigid motion) and DC-CM-002 (Coulomb law in atomic units)."""
    try:
        with _checking() as run:
            if not run:
                return
            if featurizer.randomize:
                return
            confs = _conformer_arrays(mol)
            n = mol.GetNumAtoms()
            M = np.asarray(result, float)
            if (not confs or n == 0 or n > featurizer.max_atoms or
                    M.shape != (len(confs), featurizer.max_atoms,
                                featurizer.max_atoms) or not _finite(M)):
                return
            z = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], float)
            from scipy.spatial.distance import cdist
            dmin, scale = np.inf, 1e-300
            for P in confs:
                if not _sane(P, hi=1e100):
                    return
                d = cdist(P, P)
                np.fill_diagonal(d, np.inf)
                dmin = min(dmin, float(d.min()))
                scale = max(scale, _maxabs(P))
            if not (dmin >= 1e-3):
                return
            cond = (1.0 + scale / dmin)
            # CM-002: M_ij * r_ij / a0 = Z_i Z_j  (rel. 1e-8 covers CODATA
            # revisions of a0, which differ by ~3e-11)
            rel = 1e-8 + _C * _EPS * cond
            _reach("DC-CM-002")
            zz = np.outer(z, z)
            off = ~np.eye(n, dtype=bool)
            for c, P in enumerate(confs):
                r = cdist(P, P) / _bohr()
                np.fill_diagonal(r, 1.0)
                got = M[c, :n, :n]
                exp = zz / r
                if np.any(np.abs(got - exp)[off] > rel * np.abs(exp)[off]):
                    trigger("DC-CM-002")
                    break
            # CM-001: move every conformer rigidly and recompute
            from rdkit import Chem
            from rdkit.Geometry import Point3D
            moved = Chem.Mol(mol)
            t = None
            for conf, P in zip(moved.GetConformers(), confs):
                P2, t = _rigid(P, scale)
                for i in range(n):
                    conf.SetAtomPosition(i, Point3D(*map(float, P2[i])))
            M2 = np.asarray(featurizer.coulomb_matrix(moved), float)
            tol = _C * _EPS * (scale + np.linalg.norm(t)) / dmin * max(_maxabs(M), 1e-300)
            trigger_if(M2.shape != M.shape or _violates(np.max(np.abs(M2 - M)), tol),
                       "DC-CM-001")
    except Exception:
        _swallowed()


def check_coulomb_eig(featurizer, datapoint, result):
    """DC-CM-003: permutation-invariant spectrum, sorted by |lambda|."""
    try:
        with _checking() as run:
            if not run:
                return
            if featurizer.randomize or len(datapoint.GetConformers()) == 0:
                return
            n = datapoint.GetNumAtoms()
            f = np.atleast_2d(np.asarray(result))
            if n == 0 or n > featurizer.max_atoms or not _finite(f):
                return
            from scipy.spatial.distance import cdist
            for P in _conformer_arrays(datapoint):
                d = cdist(P, P)
                np.fill_diagonal(d, np.inf)
                if not _sane(P, hi=1e100) or float(d.min()) < 1e-3:
                    return
            from rdkit import Chem
            perm = np.random.RandomState(12345).permutation(n).tolist()
            other = Chem.RenumberAtoms(Chem.Mol(datapoint), perm)
            g = np.atleast_2d(np.asarray(featurizer._featurize(other)))
            lam = np.abs(f).max()
            tol = 256 * _EPS * n * max(lam, 1e-300)
            bad = f.shape != g.shape
            if not bad:
                for a, b in zip(f, g):
                    if _violates(np.max(np.abs(np.sort(a.real) - np.sort(b.real))),
                                 tol):
                        bad = True
                    if np.any(np.diff(np.abs(a)) > tol):
                        bad = True
            trigger_if(bad, "DC-CM-003")
    except Exception:
        _swallowed()


def check_randomize_coulomb(m, result):
    """DC-CM-004: randomised matrices are simultaneous row/col permutations."""
    try:
        with _checking() as run:
            if not run:
                return
            m = np.asarray(m, float)
            if m.ndim != 2 or m.shape[0] != m.shape[1] or not _sane(m, hi=1e100):
                return
            if m.shape[0] > 300 or np.max(np.abs(m - m.T)) > 0:
                return
            tol = _C * _EPS * m.shape[0] * max(_maxabs(m), 1e-300)
            w = np.linalg.eigvalsh(m)
            _reach("DC-CM-004")
            for new in result:
                new = np.asarray(new, float)
                if new.shape != m.shape:
                    trigger("DC-CM-004")
                    return
                bad = (not np.array_equal(np.sort(new.ravel()), np.sort(m.ravel())) or
                       not np.array_equal(np.sort(np.diag(new)), np.sort(np.diag(m))) or
                       _violates(np.max(np.abs(new - new.T)), 0.0) or
                       _violates(np.max(np.abs(np.linalg.eigvalsh(new) - w)), tol))
                if bad:
                    trigger("DC-CM-004")
                    return
    except Exception:
        _swallowed()


# ======================================================== H. element constants
def check_atom_mass(atom_z, value):
    """DC-PT-001: tabulated mass lies within the known isotope-mass range."""
    try:
        with _checking() as run:
            if not run:
                return
            z = int(atom_z)
            if z != atom_z or not (1 <= z <= 118) or not math.isfinite(float(value)):
                return
            from rdkit import Chem
            from scipy import constants
            pt = Chem.GetPeriodicTable()
            masses = []
            for a in range(1, 320):
                try:
                    m = pt.GetMassForIsotope(z, a)
                except Exception:
                    continue
                if m > 0:
                    masses.append(m)
            if not masses:
                return
            ratio = (constants.physical_constants["atomic mass constant"][0] /
                     constants.physical_constants["electron mass"][0])
            u = float(value) / ratio
            lo, hi = min(masses), max(masses)
            trigger_if(not (lo * (1 - 1e-6) <= u <= hi * (1 + 1e-6)), "DC-PT-001")
    except Exception:
        _swallowed()


def _madelung_period(z):
    orbitals = sorted(((n, l) for n in range(1, 9) for l in range(0, n)),
                      key=lambda nl: (nl[0] + nl[1], nl[0]))
    left, top = z, 0
    for n, l in orbitals:
        if left <= 0:
            break
        take = min(left, 2 * (2 * l + 1))
        left -= take
        top = max(top, n)
    return top


def check_period(atom_z, period):
    """DC-PT-002: period equals the Madelung-rule outermost shell."""
    try:
        with _checking() as run:
            if not run:
                return
            if isinstance(atom_z, bool) or not isinstance(atom_z, (int, np.integer)):
                return
            if not 1 <= int(atom_z) <= 118:
                return
            trigger_if(int(period) != _madelung_period(int(atom_z)), "DC-PT-002")
    except Exception:
        _swallowed()


def check_atomz(symbol, atom_z):
    """DC-PT-003: symbol-to-Z map agrees with an independent periodic table."""
    try:
        with _checking() as run:
            if not run:
                return
            from rdkit import Chem
            try:
                ref = Chem.GetPeriodicTable().GetAtomicNumber(str(symbol))
            except Exception:
                return
            trigger_if(int(atom_z) != int(ref), "DC-PT-003")
    except Exception:
        _swallowed()


# ============================================================== I. splitters
def _valid_smiles(ids):
    from rdkit import Chem
    out = {}
    for i, s in enumerate(ids):
        m = Chem.MolFromSmiles(str(s))
        if m is not None:
            out[i] = m
    return out


def check_scaffold_split(ids, train, valid, test):
    """DC-SP-001: no Bemis-Murcko scaffold spans two splits."""
    try:
        with _checking() as run:
            if not run:
                return
            from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
            mols = _valid_smiles(ids)
            seen = {}
            for label, part in (("train", train), ("valid", valid), ("test", test)):
                for i in part:
                    if int(i) not in mols:
                        continue
                    sc = MurckoScaffoldSmiles(mol=mols[int(i)], includeChirality=False)
                    seen.setdefault(sc, set()).add(label)
            trigger_if(any(len(v) > 1 for v in seen.values()), "DC-SP-001")
    except Exception:
        _swallowed()


def check_mass_split(ids, train, valid, test):
    """DC-SP-002: exact masses are ordered train <= valid <= test."""
    try:
        with _checking() as run:
            if not run:
                return
            from rdkit.Chem import rdMolDescriptors
            mols = _valid_smiles(ids)
            mass = {i: rdMolDescriptors.CalcExactMolWt(m) for i, m in mols.items()}
            groups = [[mass[int(i)] for i in part if int(i) in mass]
                      for part in (train, valid, test)]
            groups = [g for g in groups if g]
            top = max((m for g in groups for m in g), default=1.0)
            tol = 1e-9 * max(top, 1.0)
            bad = any(max(a) > min(b) + tol for a, b in zip(groups, groups[1:]))
            trigger_if(bad, "DC-SP-002")
    except Exception:
        _swallowed()
