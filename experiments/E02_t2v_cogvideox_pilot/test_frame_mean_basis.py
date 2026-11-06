"""CPU checks of fit_basis_frame_mean.py (PROTOCOL_ko.md section 7d).

  python -m pytest experiments/E02_t2v_cogvideox_pilot/test_frame_mean_basis.py -q
"""
from pathlib import Path
import sys
import torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
from div import basis                                                         # noqa: E402
from fit_basis_frame_mean import FrameMeanMoments                             # noqa: E402
import temporal_share                                                         # noqa: E402

torch.manual_seed(0)


def test_frame_mean_moments_equal_moments_of_frame_means():
    N, F, P, C = 3, 4, 5, 6
    x = torch.randn(2*N, F*P, C)                                              # two CFG groups
    a, b = FrameMeanMoments([7], [0], N, P), basis.Moments([7], [0], N)
    a.add(7, x)
    b.add(7, torch.stack([x[:, [f*P+p for f in range(F)]].mean(1) for p in range(P)], 1))
    for k in ('D', 'C', 'H'):
        assert torch.allclose(a.acc[(0, 7)][k], b.acc[(0, 7)][k], atol=1e-10)
    assert a.acc[(0, 7)]['nD'] == 2*N*P


def test_trace_matches_temporal_share():
    """tr D (raw sum) of the frame-mean moments = sum of (d_all - d_tv) / frames of the diagnostic, and the same
    inside a basis Q with (dq_all - dq_tv): the identity the GPU check uses."""
    N, F, P, C = 2, 3, temporal_share.POSITIONS, 4
    x = torch.randn(2*N, F*P, C, dtype=torch.float64)
    Q = torch.linalg.qr(torch.randn(C, 2, dtype=torch.float64))[0]
    ts = temporal_share.TemporalShare({0: Q}, [0], 'cpu')
    ts.Q = {0: Q}
    ts.ctrl = type('Ctrl', (), dict(seeds=list(range(N)), step=0))()
    ts.add(0, x)
    m = FrameMeanMoments([0], [0], N, P)
    m.add(0, x)
    D = m.acc[(0, 0)]['D']
    want = sum(r['d_all']-r['d_tv'] for r in ts.rows)/F
    want_q = sum(r['dq_all']-r['dq_tv'] for r in ts.rows)/F
    assert abs(float(torch.trace(D))-want) < 1e-4*want
    assert abs(float(torch.trace(Q.T@D@Q))-want_q) < 1e-4*want_q


def test_frame_mean_basis_picks_frame_common_seed_directions():
    """Seeds differ along dims 4-5 by an amount common to all frames, and along dims 8-9 by a larger amount that changes
    from frame to frame (zero frame mean); prompt content lies in dims 0-1. The token basis picks 8-9, the frame-mean
    basis picks 4-5."""
    N, F, P, C, k = 6, 5, 8, 12, 2
    tok, fm = basis.Moments([0], [0], N), FrameMeanMoments([0], [0], N, P)
    for i in range(8):
        x = torch.zeros(N, F, P, C)
        x[..., 0:2] = 3*torch.randn(1, 1, P, 2)                               # prompt content (same for all seeds)
        x[..., 4:6] = torch.randn(N, 1, P, 2)                                 # seed deviation, same in every frame
        v = torch.randn(N, F, P, 2)
        x[..., 8:10] = 3*(v-v.mean(1, keepdim=True))                          # seed deviation, frame-varying
        x[..., 10:] = .1*torch.randn(N, F, P, 2)
        for mom in (tok, fm):
            mom.half = i % 2
            mom.add(0, x.reshape(N, F*P, C))
    bias = {0: torch.zeros(C)}
    for mom, dims in ((tok, [8, 9]), (fm, [4, 5])):
        Q = basis.fit(mom.state(), bias, 'cogvideox', k)['Q'][0]
        assert float(Q[dims].pow(2).sum()) > .95*k
