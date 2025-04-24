import parsac.job.gotm
import parsac.sensitivity

if __name__ == "__main__":
    # experiment = parsac.sensitivity.MVR(20)
    # experiment = parsac.sensitivity.Morris(8)
    # experiment = parsac.sensitivity.CV(100)
    experiment = parsac.sensitivity.Sobol(8)

    sim = parsac.job.gotm.Simulation("./nns_annual", executable="gotm")

    experiment.add_parameter(
        sim.get_parameter("gotm.yaml", "turbulence/turb_param/k_min"),
        1e-8,
        1e-4,
        logscale=True,
    )

    # Targets for sensitivity analysis
    sim.record_output("result.nc", "temp[:,0].max()")
    sim.record_output("result.nc", "temp[:,0].min()")
    sim.record_output("result.nc", "temp[:,-1].max()")
    sim.record_output("result.nc", "temp[:,-1].min()")

    experiment.add_job(sim)

    p = experiment.run()
