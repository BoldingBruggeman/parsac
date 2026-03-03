import parsac.job.gotm

if __name__ == "__main__":
    experiment = parsac.core.Ensemble()

    sim = parsac.job.gotm.Simulation("./nns_annual", executable="gotm")

    experiment.add_parameter(
        sim.get_parameter("gotm.yaml", "turbulence/turb_param/k_min"),
        1e-8,
        1e-4,
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

    experiment.add_job(sim)

    ensemble_members = []
    work_dirs = []
    for k_min in [1e-7, 1e-6, 1e-5]:
        for u10_scale in [0.8, 1.0, 1.2]:
            for v10_scale in [0.8, 1.0, 1.2]:
                work_dirs.append(f"member{len(ensemble_members):03}")
                ensemble_members.append([k_min, u10_scale, v10_scale])

    experiment.run(ensemble_members, work_dirs)
