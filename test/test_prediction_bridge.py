"""Unit tests for core/prediction_bridge.py"""
import os, sys
import pytest, torch
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from core.prediction_bridge import (
    _normalise_pred_type, _snr_from_flowmatch_sigma, _snr_from_ddpm_alpha_bar,
    _ddpm_timesteps_matching_fm_sigma, _fm_timesteps_matching_ddpm_timestep,
    _recover_x0_from_epsilon, _recover_x0_from_vpred, _recover_x0_from_fm_velocity,
    _x0_to_epsilon, _x0_to_vpred, _x0_to_fm_velocity,
    get_prediction_bridge,
    IdentityBridge, VPredToFMBridge, EpsilonToFMBridge,
    FMToVPredBridge, FMToEpsilonBridge, EpsilonToVPredBridge, VPredToEpsilonBridge,
)
# ── Mock schedulers ────────────────────────────────────────────────────────────
class MockDDPMScheduler:
    def __init__(self, prediction_type="epsilon", num_timesteps=1000):
        betas = torch.linspace(0.00085, 0.012, num_timesteps)
        self.alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
        class _C:
            pass
        self.config = _C()
        self.config.prediction_type = prediction_type
        self.config.num_train_timesteps = num_timesteps
    def add_noise(self, latents, noise, timesteps):
        ac = self.alphas_cumprod[timesteps]
        s = ac.sqrt().view(-1,1,1,1)
        r = (1-ac).sqrt().view(-1,1,1,1)
        return s*latents + r*noise
class MockFMScheduler:
    def __init__(self, num_timesteps=1000):
        N = num_timesteps
        self.timesteps = torch.arange(N, 0, -1, dtype=torch.float32)
        self.sigmas    = self.timesteps / N
        class _C:
            pass
        self.config = _C()
        self.config.prediction_type = "flow_prediction"
        self.config.num_train_timesteps = N
    def get_sigmas_for_timesteps(self, timesteps):
        match = (timesteps.cpu().unsqueeze(-1) == self.timesteps.unsqueeze(0))
        idx   = match.float().argmax(dim=-1)
        return self.sigmas[idx].to(timesteps.device)
    def scale_noise(self, latents, timesteps, noise):
        sig = self.get_sigmas_for_timesteps(timesteps).view(-1,1,1,1)
        return (1-sig)*latents + sig*noise
    def add_noise(self, latents, noise, timesteps):
        return self.scale_noise(latents, timesteps, noise)
# ── Fixtures ───────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def ddpm_eps():   return MockDDPMScheduler("epsilon")
@pytest.fixture(scope="module")
def ddpm_vpred(): return MockDDPMScheduler("v_prediction")
@pytest.fixture(scope="module")
def fm():         return MockFMScheduler()
@pytest.fixture(scope="module")
def latent_batch():
    torch.manual_seed(42)
    return torch.randn(2,4,8,8), torch.randn(2,4,8,8)
# ── Helpers ────────────────────────────────────────────────────────────────────
def _gt_fm_target(x1, eps):      return eps - x1
def _gt_eps_target(eps):         return eps
def _gt_vpred(sched, x1, eps, ts):
    ac = sched.alphas_cumprod[ts]
    s  = ac.sqrt().view(-1,1,1,1); r = (1-ac).sqrt().view(-1,1,1,1)
    return s*eps - r*x1
def _perfect_eps(x1, eps):   return eps
def _perfect_vpred(sched, x1, eps, ts):  return _gt_vpred(sched, x1, eps, ts)
def _perfect_fm_vel(x1, eps): return eps - x1
def _ddpm_noisy(sched, x1, eps, ts):
    ac = sched.alphas_cumprod[ts]
    s  = ac.sqrt().view(-1,1,1,1); r = (1-ac).sqrt().view(-1,1,1,1)
    return s*x1 + r*eps
# ── _normalise_pred_type ───────────────────────────────────────────────────────
@pytest.mark.parametrize("s,expected", [
    ("epsilon","epsilon"),("v_prediction","v_prediction"),
    ("v-prediction","v_prediction"),("flow_prediction","flow_prediction"),
    ("flow-matching","flow_prediction"),("flow_match","flow_prediction"),
])
def test_normalise_valid(s, expected):
    assert _normalise_pred_type(s) == expected
def test_normalise_invalid():
    with pytest.raises(ValueError): _normalise_pred_type("bogus")
# ── Factory ────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("t,s,cls", [
    ("epsilon",         "epsilon",         IdentityBridge),
    ("v_prediction",    "v_prediction",    IdentityBridge),
    ("flow_prediction", "flow_prediction", IdentityBridge),
    ("v_prediction",    "flow_prediction", VPredToFMBridge),
    ("epsilon",         "flow_prediction", EpsilonToFMBridge),
    ("flow_prediction", "v_prediction",    FMToVPredBridge),
    ("flow_prediction", "epsilon",         FMToEpsilonBridge),
    ("epsilon",         "v_prediction",    EpsilonToVPredBridge),
    ("v_prediction",    "epsilon",         VPredToEpsilonBridge),
])
def test_factory_pairs(t, s, cls):
    assert isinstance(get_prediction_bridge(t, s), cls)
def test_factory_unsupported():
    with pytest.raises(ValueError): get_prediction_bridge("flow_prediction","unknown")
# ── SNR invertibility ──────────────────────────────────────────────────────────
@pytest.mark.parametrize("sigma",[0.05,0.2,0.5,0.8,0.95])
def test_snr_helpers_roundtrip(sigma):
    sig_t  = torch.tensor([sigma])
    snr_fm = _snr_from_flowmatch_sigma(sig_t)
    ab     = snr_fm / (1+snr_fm)
    snr_d  = _snr_from_ddpm_alpha_bar(ab)
    sig_rc = 1.0 / (1.0 + snr_d.sqrt())
    assert abs(sig_rc.item() - sigma) < 1e-5
# ── SNR-matching functions ─────────────────────────────────────────────────────
def test_ddpm_ts_matching_fm_sigma(ddpm_eps):
    sigmas  = torch.tensor([0.1,0.3,0.5,0.7,0.9])
    ddpm_ts = _ddpm_timesteps_matching_fm_sigma(sigmas, ddpm_eps)
    for s, t in zip(sigmas, ddpm_ts):
        snr = ((1-s)/s)**2
        ab_target = snr/(1+snr)
        ab_actual = ddpm_eps.alphas_cumprod[t.long()]
        assert abs(ab_actual.item()-ab_target.item()) < 0.02
def test_fm_ts_matching_ddpm_ts(ddpm_eps, fm):
    ddpm_ts = torch.tensor([100,300,500,700,900])
    fm_ts   = _fm_timesteps_matching_ddpm_timestep(ddpm_ts, ddpm_eps, fm)
    for t_d, t_fm in zip(ddpm_ts, fm_ts):
        sig      = fm.get_sigmas_for_timesteps(t_fm.unsqueeze(0)).item()
        snr_fm   = ((1-sig)/(sig+1e-8))**2
        ab_fm    = snr_fm/(1+snr_fm)
        ab_ddpm  = ddpm_eps.alphas_cumprod[t_d.long()].item()
        assert abs(ab_fm-ab_ddpm) < 0.05
# ── x0 recovery ───────────────────────────────────────────────────────────────
@pytest.mark.parametrize("t",[100,500,900])
def test_recover_x0_from_epsilon(ddpm_eps, latent_batch, t):
    x1,eps = latent_batch; ts=torch.tensor([t,t])
    x_t = ddpm_eps.add_noise(x1,eps,ts)
    ab  = ddpm_eps.alphas_cumprod[ts]
    x0  = _recover_x0_from_epsilon(x_t.float(),eps.float(),ab.float())
    assert torch.allclose(x0,x1.float(),atol=1e-5), f"t={t} err={(x0-x1.float()).abs().max():.2e}"
@pytest.mark.parametrize("t",[100,500,900])
def test_recover_x0_from_vpred(ddpm_vpred, latent_batch, t):
    x1,eps = latent_batch; ts=torch.tensor([t,t])
    ab  = ddpm_vpred.alphas_cumprod[ts]
    s=ab.sqrt().view(-1,1,1,1); r=(1-ab).sqrt().view(-1,1,1,1)
    v   = s*eps - r*x1
    x_t = s*x1 + r*eps
    x0  = _recover_x0_from_vpred(x_t.float(),v.float(),ab.float())
    assert torch.allclose(x0,x1.float(),atol=1e-5), f"t={t} err={(x0-x1.float()).abs().max():.2e}"
@pytest.mark.parametrize("tv",[100.0,500.0,900.0])
def test_recover_x0_from_fm_velocity(fm, latent_batch, tv):
    x1,eps = latent_batch; ts=torch.tensor([tv,tv])
    sig = fm.get_sigmas_for_timesteps(ts).view(-1,1,1,1)
    x_t = (1-sig)*x1 + sig*eps
    v   = eps - x1
    x0  = _recover_x0_from_fm_velocity(x_t.float(),v.float(),sig.squeeze(-1).squeeze(-1).squeeze(-1).float())
    assert torch.allclose(x0,x1.float(),atol=1e-5), f"tv={tv} err={(x0-x1.float()).abs().max():.2e}"
# ── IdentityBridge ─────────────────────────────────────────────────────────────
def test_identity_bridge(ddpm_eps, latent_batch):
    x1,eps = latent_batch; ts=torch.tensor([200,500])
    b = IdentityBridge()
    out = torch.randn_like(x1)
    assert torch.allclose(b.convert_output(out,x1,ts,ts,eps,ddpm_eps,ddpm_eps), out)
# ── EpsilonToVPredBridge ───────────────────────────────────────────────────────
@pytest.mark.parametrize("t",[100,500,900])
def test_epsilon_to_vpred(ddpm_eps, ddpm_vpred, latent_batch, t):
    x1,eps = latent_batch; ts=torch.tensor([t,t])
    b = EpsilonToVPredBridge()
    tts = b.remap_timesteps(ts,ddpm_eps,ddpm_vpred)
    x_t = b.build_noisy_latents(x1,eps,tts,ddpm_eps)
    out = b.convert_output(_perfect_eps(x1,eps),x_t,tts,ts,eps,ddpm_eps,ddpm_vpred)
    gt  = _gt_vpred(ddpm_vpred,x1,eps,ts)
    assert torch.allclose(out,gt.float(),atol=1e-5), f"t={t} err={(out-gt.float()).abs().max():.2e}"
# ── VPredToEpsilonBridge ───────────────────────────────────────────────────────
@pytest.mark.parametrize("t",[100,500,900])
def test_vpred_to_epsilon(ddpm_eps, ddpm_vpred, latent_batch, t):
    x1,eps = latent_batch; ts=torch.tensor([t,t])
    b = VPredToEpsilonBridge()
    tts = b.remap_timesteps(ts,ddpm_vpred,ddpm_eps)
    x_t = b.build_noisy_latents(x1,eps,tts,ddpm_vpred)
    out = b.convert_output(_perfect_vpred(ddpm_vpred,x1,eps,tts),x_t,tts,ts,eps,ddpm_vpred,ddpm_eps)
    gt  = _gt_eps_target(eps)
    assert torch.allclose(out,gt.float(),atol=1e-5), f"t={t} err={(out-gt.float()).abs().max():.2e}"
# ── VPredToFMBridge ────────────────────────────────────────────────────────────
@pytest.mark.parametrize("tv",[100.0,500.0,800.0])
def test_vpred_to_fm(ddpm_vpred, fm, latent_batch, tv):
    x1,eps = latent_batch; sts=torch.tensor([tv,tv])
    b = VPredToFMBridge()
    tts = b.remap_timesteps(sts,fm,ddpm_vpred)
    x_t = b.build_noisy_latents(x1,eps,tts,ddpm_vpred)
    out = b.convert_output(_perfect_vpred(ddpm_vpred,x1,eps,tts),x_t,tts,sts,eps,ddpm_vpred,fm)
    gt  = _gt_fm_target(x1,eps)
    assert torch.allclose(out,gt.float(),atol=1e-5), f"tv={tv} err={(out-gt.float()).abs().max():.2e}"
# ── EpsilonToFMBridge ──────────────────────────────────────────────────────────
@pytest.mark.parametrize("tv",[100.0,500.0,800.0])
def test_epsilon_to_fm(ddpm_eps, fm, latent_batch, tv):
    x1,eps = latent_batch; sts=torch.tensor([tv,tv])
    b = EpsilonToFMBridge()
    tts = b.remap_timesteps(sts,fm,ddpm_eps)
    x_t = b.build_noisy_latents(x1,eps,tts,ddpm_eps)
    out = b.convert_output(_perfect_eps(x1,eps),x_t,tts,sts,eps,ddpm_eps,fm)
    gt  = _gt_fm_target(x1,eps)
    assert torch.allclose(out,gt.float(),atol=1e-5), f"tv={tv} err={(out-gt.float()).abs().max():.2e}"
# ── FMToVPredBridge ────────────────────────────────────────────────────────────
@pytest.mark.parametrize("t",[100,500,900])
def test_fm_to_vpred(ddpm_vpred, fm, latent_batch, t):
    x1,eps = latent_batch; sts=torch.tensor([t,t])
    b = FMToVPredBridge()
    tts = b.remap_timesteps(sts,ddpm_vpred,fm)
    x_t = b.build_noisy_latents(x1,eps,tts,fm)
    out = b.convert_output(_perfect_fm_vel(x1,eps),x_t,tts,sts,eps,fm,ddpm_vpred)
    gt  = _gt_vpred(ddpm_vpred,x1,eps,sts)
    assert torch.allclose(out,gt.float(),atol=1e-5), f"t={t} err={(out-gt.float()).abs().max():.2e}"
# ── FMToEpsilonBridge ──────────────────────────────────────────────────────────
@pytest.mark.parametrize("t",[100,500,900])
def test_fm_to_epsilon(ddpm_eps, fm, latent_batch, t):
    x1,eps = latent_batch; sts=torch.tensor([t,t])
    b = FMToEpsilonBridge()
    tts = b.remap_timesteps(sts,ddpm_eps,fm)
    x_t = b.build_noisy_latents(x1,eps,tts,fm)
    out = b.convert_output(_perfect_fm_vel(x1,eps),x_t,tts,sts,eps,fm,ddpm_eps)
    gt  = _gt_eps_target(eps)
    assert torch.allclose(out,gt.float(),atol=1e-5), f"t={t} err={(out-gt.float()).abs().max():.2e}"
# ── SNR roundtrip (remap forward then back) ────────────────────────────────────
def test_snr_roundtrip_vpred_fm(ddpm_vpred, fm):
    fm_ts = torch.tensor([100.0,300.0,500.0,700.0,900.0])
    ddpm_ts = VPredToFMBridge().remap_timesteps(fm_ts, fm, ddpm_vpred)
    fm_back = FMToVPredBridge().remap_timesteps(ddpm_ts, ddpm_vpred, fm)
    for orig, back in zip(fm_ts, fm_back):
        s_o = fm.get_sigmas_for_timesteps(orig.unsqueeze(0)).item()
        s_b = fm.get_sigmas_for_timesteps(back.unsqueeze(0)).item()
        assert abs(s_o-s_b) < 0.015, f"fm_ts={orig:.0f}: σ_orig={s_o:.4f} σ_back={s_b:.4f}"
def test_snr_roundtrip_eps_fm(ddpm_eps, fm):
    fm_ts = torch.tensor([100.0,300.0,500.0,700.0,900.0])
    ddpm_ts = EpsilonToFMBridge().remap_timesteps(fm_ts, fm, ddpm_eps)
    fm_back = FMToEpsilonBridge().remap_timesteps(ddpm_ts, ddpm_eps, fm)
    for orig, back in zip(fm_ts, fm_back):
        s_o = fm.get_sigmas_for_timesteps(orig.unsqueeze(0)).item()
        s_b = fm.get_sigmas_for_timesteps(back.unsqueeze(0)).item()
        assert abs(s_o-s_b) < 0.015, f"fm_ts={orig:.0f}: σ_orig={s_o:.4f} σ_back={s_b:.4f}"
