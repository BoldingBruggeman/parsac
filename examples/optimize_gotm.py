import parsac.job.gotm
import parsac.optimize
from parsac.util import TextFormat

if __name__ == "__main__":
    experiment = parsac.optimize.Optimization()

    sim = parsac.job.gotm.Simulation("./nns_annual", executable="gotm")

    experiment.add_parameter(
        sim.get_parameter("gotm.yaml", "turbulence/turb_param/k_min"),
        1e-8,
        1e-4,
        logscale=True,
    )

    experiment.add_target(
        sim.request_comparison(
            "result.nc",
            "temp[:,-1]",
            "./nns_annual/cci_sst.dat",
            obs_file_format=TextFormat.DEPTH_INDEPENDENT,
        )
    )

    p = experiment.run(reltol=0.00001)
