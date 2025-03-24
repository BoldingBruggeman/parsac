import parsac.job.gotm
import parsac.optimize
from parsac.util import TextFormat

job = parsac.optimize.Optimization()

sim = parsac.job.gotm.Simulation("./nns_annual", executable="gotm")

job.add_parameter(
    sim.get_parameter("gotm.yaml", "turbulence/turb_param/k_min"),
    1e-8,
    1e-4,
    logscale=True,
)

# Estimate a single scale factor for both wind components
u10_scale = sim.get_parameter("gotm.yaml", "surface/u10/scale_factor", default=1.0)
v10_scale = sim.get_parameter("gotm.yaml", "surface/v10/scale_factor", default=1.0)
job.add_parameter(u10_scale, 0.0, 2.0)
v10_scale.infer(lambda x: x, u10_scale)

# Record additional diagnostics for each model evaluation
sim.record_output("result.nc", "temp[:,0].max()")
sim.record_output("result.nc", "temp[:,0].min()")
sim.record_output("result.nc", "temp[:,-1].max()")
sim.record_output("result.nc", "temp[:,-1].min()")

job.add_target(
    sim.request_comparison(
        "result.nc",
        "temp[:,-1]",
        "./nns_annual/cci_sst.dat",
        obs_file_format=TextFormat.DEPTH_INDEPENDENT,
    ),
)

if __name__ == "__main__":
    p = job.run(reltol=0.00001)
