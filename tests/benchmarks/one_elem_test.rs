use fe_engine::prelude::*;

#[test]
fn test_single_element_point_load() {
    let mut builder = ModelBuilder::new("One Element");
    
    let mat = Material {
        id: 0,
        name: "Steel".to_string(),
        material_type: MaterialType::Steel,
        elastic_modulus: 210e9,
        poisson_ratio: 0.3,
        density: 7850.0,
        thermal_expansion: 12e-6,
        code_reference: None,
    };
    builder.add_material(mat);
    
    let section = Section {
        area: 0.01,
        inertia_y: 1.0e-5,
        inertia_z: 1.0e-5,
        torsion_constant: 1.0e-6,
    };
    
    let n0 = builder.add_node(Point3D { x: 0.0, y: 0.0, z: 0.0 });
    let n1 = builder.add_node(Point3D { x: 1.0, y: 0.0, z: 0.0 });
    
    builder.add_beam_element(n0, n1, 0, section.clone()).unwrap();
    
    // Simply supported
    builder.add_support(n0, SupportType::Pinned).unwrap();
    builder.add_support(n1, SupportType::Pinned).unwrap();
    
    // 1000 N downward at midspan - but we only have 1 element, so apply to node 1
    builder.create_load_case("Point", LoadType::Live)
        .add_element_load(
            0,  // element ID
            LoadDistribution::Concentrated {
                node_id: n1,
                force: Vector3D { x: 0.0, y: 0.0, z: -1000.0 },
                moment: None,
            }
        )
        .unwrap()
        .finish();
    
    let model = builder.build().unwrap();
    
    let solver = CpuCholesky;
    let load_case = &model.load_cases[0];
    
    let mut pipeline = AnalysisPipeline::new(&model);
    let result = pipeline.run(&solver, load_case).unwrap();
    
    let disp_n1 = result.displacement_at_node(n1).unwrap();
    println!("\nSingle element test:");
    println!("  FEA deflection: {:.6e} m", disp_n1.translation.z.abs());
    
    // For a simply-supported beam with point load P at center:
    // δ_max = PL³/(48EI)
    let e = 210e9;
    let i = 1.0e-5;
    let l = 1.0_f64;
    let p = 1000.0;
    let expected = p * l.powi(3) / (48.0 * e * i);
    
    println!("  Analytical:     {:.6e} m", expected);
    println!("  FEA/Analytical: {:.3}", disp_n1.translation.z.abs() / expected);
}
