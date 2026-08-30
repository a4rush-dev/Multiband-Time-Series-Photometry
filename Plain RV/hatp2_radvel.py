# NOTE: ignoring hat-p-2c, the second planet. Not fitting quadratic trend for it either.
# NOTE: check time units when using in other scripts

import os
import numpy as np
import radvel
import radvel.likelihood as rlike

import emcee

# NOTE: Python 3.13 sometimes has issues, use <3.10 when needed.
# Make RadVel happy on Windows
if "HOME" not in os.environ:
    os.environ["HOME"] = os.environ.get("USERPROFILE", os.path.expanduser("~"))

def e_omega_from_se(secosw_val, sesinw_val):
    e_val = secosw_val**2 + sesinw_val**2
    omega_val = np.arctan2(sesinw_val, secosw_val)
    return e_val, omega_val

def phase_fold(t, tc_use, per_use):
    return np.mod((t - tc_use) / per_use, 1.0)

class HatP2RVModel:
    def __init__(self, rv_file="hat_p2_rv.txt"):
        data = np.loadtxt(rv_file)
        t_raw = data[:, 0]
        rv_all = data[:, 1]
        err_all = data[:, 2]
        inst_id = data[:, 3].astype(int)

        t_all = t_raw + 2450000.0

        self.t_all = t_all
        self.rv_all = rv_all
        self.err_all = err_all
        self.inst_id = inst_id

        self.mask_hires = (inst_id == 0)
        self.mask_harpsn = (inst_id == 1)

        # Time base for possible trends (not used currently, still kept)
        self.time_base = np.median(self.t_all)
        self.t_rel = self.t_all - self.time_base

        self.per0 = 5.6334729      # days, orbital period
        self.tc0 = 2454529.674     # BJD_TDB, reference transit time
        self.e0 = 0.502            # eccentricity
        wdeg0 = 188.8              # degrees
        self.wrad0 = np.deg2rad(wdeg0)

        self.secosw0 = np.sqrt(self.e0) * np.cos(self.wrad0)
        self.sesinw0 = np.sqrt(self.e0) * np.sin(self.wrad0)

        self.ln_jit_center = np.log(5.0)
        self.ln_jit_sigma = 0.7

        params = radvel.Parameters(1, basis="per tc secosw sesinw k")
        params["per1"] = radvel.Parameter(value=self.per0, vary=False)
        params["tc1"] = radvel.Parameter(value=self.tc0, vary=True)
        params["secosw1"] = radvel.Parameter(value=self.secosw0, vary=True)
        params["sesinw1"] = radvel.Parameter(value=self.sesinw0, vary=True)
        params["k1"] = radvel.Parameter(value=950.0, vary=True)

        params["gamma_hires"] = radvel.Parameter(
            value=np.mean(self.rv_all[self.mask_hires]),
            vary=True,
            linear=True
        )
        params["jit_hires"] = radvel.Parameter(value=3.0, vary=True)

        params["gamma_harpsn"] = radvel.Parameter(
            value=np.mean(self.rv_all[self.mask_harpsn]),
            vary=True,
            linear=True
        )
        params["jit_harpsn"] = radvel.Parameter(value=3.0, vary=True)

        model = radvel.RVModel(params)
        model.time_base = self.time_base

        self.params = params
        self.model = model

        self._initialize_with_maxlike()

    def _initialize_with_maxlike(self):
        like_hires = rlike.RVLikelihood(
            self.model,
            self.t_all[self.mask_hires],
            self.rv_all[self.mask_hires],
            self.err_all[self.mask_hires],
            suffix="_hires",
        )
        like_harpsn = rlike.RVLikelihood(
            self.model,
            self.t_all[self.mask_harpsn],
            self.rv_all[self.mask_harpsn],
            self.err_all[self.mask_harpsn],
            suffix="_harpsn",
        )

        like_hires.params["gamma_hires"] = self.params["gamma_hires"]
        like_hires.params["jit_hires"] = self.params["jit_hires"]

        like_harpsn.params["gamma_harpsn"] = self.params["gamma_harpsn"]
        like_harpsn.params["jit_harpsn"] = self.params["jit_harpsn"]

        comp_like = rlike.CompositeLikelihood([like_hires, like_harpsn])

        comp_like = radvel.fitting.maxlike_fitting(comp_like)
        best = comp_like.params

        self.params["tc1"].value = best["tc1"].value
        self.params["k1"].value = best["k1"].value
        self.params["gamma_hires"].value = best["gamma_hires"].value
        self.params["gamma_harpsn"].value = best["gamma_harpsn"].value
        self.params["jit_hires"].value = best["jit_hires"].value
        self.params["jit_harpsn"].value = best["jit_harpsn"].value

        self.tc_best = self.params["tc1"].value
        self.k_best = self.params["k1"].value

    def theta_from_params(self):
        tc1 = self.params["tc1"].value
        k1 = self.params["k1"].value
        secosw1 = self.params["secosw1"].value
        sesinw1 = self.params["sesinw1"].value
        gamma_hires = self.params["gamma_hires"].value
        gamma_harpsn = self.params["gamma_harpsn"].value
        ln_jit_hires = np.log(self.params["jit_hires"].value)
        ln_jit_harpsn = np.log(self.params["jit_harpsn"].value)

        return np.array([
            tc1, k1, secosw1, sesinw1,
            gamma_hires, gamma_harpsn,
            ln_jit_hires, ln_jit_harpsn,
        ])

    def update_params_from_theta(self, theta):
        (tc1, k1, secosw1, sesinw1,
         gamma_hires, gamma_harpsn,
         ln_jit_hires, ln_jit_harpsn) = theta

        self.params["tc1"].value = tc1
        self.params["k1"].value = k1
        self.params["secosw1"].value = secosw1
        self.params["sesinw1"].value = sesinw1
        self.params["gamma_hires"].value = gamma_hires
        self.params["gamma_harpsn"].value = gamma_harpsn
        self.params["jit_hires"].value = np.exp(ln_jit_hires)
        self.params["jit_harpsn"].value = np.exp(ln_jit_harpsn)

    def log_prior(self, theta):
        (tc1, k1, secosw1, sesinw1,
         gamma_hires, gamma_harpsn,
         ln_jit_hires, ln_jit_harpsn) = theta

        if not (self.tc0 - 0.5 < tc1 < self.tc0 + 0.5):
            return -np.inf
        if not (500.0 < k1 < 1500.0):
            return -np.inf
        if not (-2000.0 < gamma_hires < 2000.0):
            return -np.inf
        if not (-2000.0 < gamma_harpsn < 2000.0):
            return -np.inf
        if not (np.log(0.5) < ln_jit_hires < np.log(30.0)):
            return -np.inf
        if not (np.log(0.5) < ln_jit_harpsn < np.log(30.0)):
            return -np.inf

        e_val, omega_val = e_omega_from_se(secosw1, sesinw1)
        if not (0.0 < e_val < 0.999):
            return -np.inf

        sig_tc = 0.02
        sig_K = 50.0
        sig_e = 0.02
        sig_w = np.deg2rad(5.0)

        lp = 0.0

        lp += -0.5 * ((tc1 - self.tc0)**2 / sig_tc**2
                     + np.log(2.0 * np.pi * sig_tc**2))
        
        lp += -0.5 * ((k1 - self.k_best)**2 / sig_K**2 + np.log(2.0 * np.pi * sig_K**2))
        
        lp += -0.5 * ((e_val - self.e0)**2 / sig_e**2 + np.log(2.0 * np.pi * sig_e**2))
        
        lp += -0.5 * ((omega_val - self.wrad0)**2 / sig_w**2
                     + np.log(2.0 * np.pi * sig_w**2))

        lp += -0.5 * ((ln_jit_hires - self.ln_jit_center)**2
                     / self.ln_jit_sigma**2 + np.log(2.0 * np.pi * self.ln_jit_sigma**2))
        
        lp += -0.5 * ((ln_jit_harpsn - self.ln_jit_center)**2
                     / self.ln_jit_sigma**2
                     + np.log(2.0 * np.pi * self.ln_jit_sigma**2))

        return lp

    def log_likelihood(self, theta):
        self.update_params_from_theta(theta)

        rv_planet = self.model(self.t_all)

        (tc1, k1, secosw1, sesinw1,
         gamma_hires, gamma_harpsn,
         ln_jit_hires, ln_jit_harpsn) = theta

        jit_hires = np.exp(ln_jit_hires)
        jit_harpsn = np.exp(ln_jit_harpsn)

        rv_model = rv_planet.copy()
        rv_model[self.mask_hires] += gamma_hires
        rv_model[self.mask_harpsn] += gamma_harpsn

        sigma_hires = np.sqrt(self.err_all[self.mask_hires]**2 + jit_hires**2)
        sigma_harpsn = np.sqrt(self.err_all[self.mask_harpsn]**2 + jit_harpsn**2)

        resid_hires = self.rv_all[self.mask_hires] - rv_model[self.mask_hires]
        resid_harpsn = self.rv_all[self.mask_harpsn] - rv_model[self.mask_harpsn]

        lnL_hires = -0.5 * np.sum(
            (resid_hires / sigma_hires)**2
            + np.log(2.0 * np.pi * sigma_hires**2)
        )
        lnL_harpsn = -0.5 * np.sum(
            (resid_harpsn / sigma_harpsn)**2
            + np.log(2.0 * np.pi * sigma_harpsn**2)
        )

        return lnL_hires + lnL_harpsn

    def log_prob(self, theta): # combined log_probability to use in other pipelines
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(theta)
    
    def run_emcee(self, nwalkers=64, nburn=4000, nprod=8000):
        theta_start = self.theta_from_params()
        ndim = len(theta_start)

        jump = np.array([1e-3, 5.0, 1e-3, 1e-3, 5.0, 5.0, 0.1, 0.1])
        pos0 = theta_start + jump * np.random.randn(nwalkers, ndim)

        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_prob)

        sampler.run_mcmc(pos0, nburn, progress=True)
        sampler.reset()

        sampler.run_mcmc(None, nprod, progress=True)

        flat = sampler.get_chain(thin=10, flat=True)
        theta_med = np.percentile(flat, 50, axis=0)
        self.update_params_from_theta(theta_med)

        return sampler, flat, theta_med

    def compute_diagnostics(self, theta=None):
        if theta is not None:
            self.update_params_from_theta(theta)

        rv_planet = self.model(self.t_all)

        gamma_hires = self.params["gamma_hires"].value
        gamma_harpsn = self.params["gamma_harpsn"].value
        jit_hires = self.params["jit_hires"].value
        jit_harpsn = self.params["jit_harpsn"].value

        rv_model = rv_planet.copy()
        rv_model[self.mask_hires] += gamma_hires
        rv_model[self.mask_harpsn] += gamma_harpsn

        resid = self.rv_all - rv_model

        sigma_hires = np.sqrt(self.err_all[self.mask_hires]**2 + jit_hires**2)
        sigma_harpsn = np.sqrt(self.err_all[self.mask_harpsn]**2 + jit_harpsn**2)

        chi2 = (
            np.sum((resid[self.mask_hires] / sigma_hires)**2)
            + np.sum((resid[self.mask_harpsn] / sigma_harpsn)**2)
        )
        dof = len(self.t_all) - 8  # 8 free parameters in theta
        chi2r = chi2 / dof
        rms = np.std(resid)

        # TODO: add bic, aic. fix offset issue in final plots

        e_val, omega_val = e_omega_from_se(
            self.params["secosw1"].value,
            self.params["sesinw1"].value,
        )

        return {
            "chi2": chi2,
            "chi2r": chi2r,
            "rms": rms,
            "e": e_val,
            "omega_rad": omega_val,
            "omega_deg": np.rad2deg(omega_val),
        }
    def get_model_and_residuals(self, theta=None):
        if theta is not None:
            self.update_params_from_theta(theta)

        rv_planet = self.model(self.t_all)

        gamma_hires = self.params["gamma_hires"].value
        gamma_harpsn = self.params["gamma_harpsn"].value

        rv_model = rv_planet.copy()
        rv_model[self.mask_hires] += gamma_hires
        rv_model[self.mask_harpsn] += gamma_harpsn

        resid = self.rv_all - rv_model

        return self.t_all, rv_model, resid

    def plot_phase_rv(self, theta_rv=None, filename=None):
        import matplotlib.pyplot as plt

        if theta_rv is not None:
            self.update_params_from_theta(theta_rv)

        t_all = self.t_all
        rv_all = self.rv_all
        err_all = self.err_all
        inst_id = self.inst_id

        rv_planet_func = self.model

        tc_use = self.params["tc1"].value
        per_use = self.per0

        phase = phase_fold(t_all, tc_use, per_use)

        order = np.argsort(phase)
        phase_sorted = phase[order]
        rv_sorted = rv_all[order]
        err_sorted = err_all[order]
        inst_sorted = inst_id[order]

        rv_planet_data = rv_planet_func(t_all)
        gamma_for_curve = self.params["gamma_hires"].value
        rv_model_data = rv_planet_data + gamma_for_curve
        resid = rv_all - rv_model_data
        resid_sorted = resid[order]

        phase_model = np.linspace(0.0, 1.0, 800)
        t_model_phase = tc_use + phase_model * per_use
        rv_planet_model = rv_planet_func(t_model_phase)
        rv_curve = rv_planet_model + gamma_for_curve

        mask_hires = (inst_sorted == 0)
        mask_harpsn = (inst_sorted == 1)

        fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

        ax[0].errorbar(
            phase_sorted[mask_hires],
            rv_sorted[mask_hires],
            yerr=err_sorted[mask_hires],
            fmt="o",
            ms=4,
            alpha=0.7,
            label="HIRES",
        )
        ax[0].errorbar(
            phase_sorted[mask_harpsn],
            rv_sorted[mask_harpsn],
            yerr=err_sorted[mask_harpsn],
            fmt="s",
            ms=4,
            alpha=0.7,
            label="HARPS-N",
        )
        ax[0].plot(
            phase_model,
            rv_curve,
            "k-",
            lw=1.7,
            label="e/ω-free one-planet median model",
        )
        ax[0].set_ylabel("RV (m/s)")
        ax[0].set_title("HAT-P-2b: phase-folded RV (e/ω-free one-planet)")
        ax[0].legend()
        ax[0].grid(alpha=0.3)

        ax[1].errorbar(
            phase_sorted[mask_hires],
            resid_sorted[mask_hires],
            yerr=err_sorted[mask_hires],
            fmt="o",
            ms=4,
            alpha=0.7,
        )
        ax[1].errorbar(
            phase_sorted[mask_harpsn],
            resid_sorted[mask_harpsn],
            yerr=err_sorted[mask_harpsn],
            fmt="s",
            ms=4,
            alpha=0.7,
        )
        ax[1].axhline(0.0, color="r", ls="--", lw=1)
        ax[1].set_xlabel("Orbital phase (cycles)")
        ax[1].set_ylabel("Residuals (m/s)")
        ax[1].grid(alpha=0.3)

        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename, dpi=200)
            plt.close(fig)
        else:
            plt.show()


    def plot_residual_histogram(self, theta=None, bins=30, filename=None):
        import matplotlib.pyplot as plt

        _, _, resid = self.get_model_and_residuals(theta=theta)

        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.hist(resid, bins=bins, histtype="stepfilled", alpha=0.7)
        ax.set_xlabel("RV residual (m/s)")
        ax.set_ylabel("Count")
        ax.set_title("HAT-P-2b RV residual distribution")
        ax.grid(alpha=0.3)

        if filename is not None:
            plt.savefig(filename, dpi=200)
            plt.close(fig)
        else:
            plt.show()