import parsac.job.gotm
import parsac.sensitivity

if __name__ == "__main__":
    # experiment = parsac.sensitivity.MVR(50)
    # experiment = parsac.sensitivity.Morris(8)
    experiment = parsac.sensitivity.Sobol(8)

    sim = parsac.job.gotm.Simulation("./nns_annual", executable="gotm")

    experiment.add_parameter(
        sim.get_parameter("gotm.yaml", "turbulence/turb_param/k_min"),
        1e-8,
        1e-4,
        logscale=True,
    )
    experiment.add_parameter(
        sim.get_parameter("gotm.yaml", "surface/u10/scale_factor", default=1.0),
        0.0,
        2.0,
    )
    experiment.add_parameter(
        sim.get_parameter("gotm.yaml", "surface/v10/scale_factor", default=1.0),
        0.0,
        2.0,
    )
    experiment.add_parameter("dummy", 0.0, 1.0)

    # Targets for sensitivity analysis:
    # minimum and maximum temperature at bottom (k=0) and surface (k=-1)
    sim.record_output("result.nc", "temp[:,0].max()")
    sim.record_output("result.nc", "temp[:,0].min()")
    sim.record_output("result.nc", "temp[:,-1].max()")
    sim.record_output("result.nc", "temp[:,-1].min()")

    experiment.add_job(sim)

    experiment.run()
