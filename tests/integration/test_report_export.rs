use fe_engine::prelude::*;
use tempfile::TempDir;

#[test]
fn test_export_displacements_csv() {
    let model = create_simple_beam_model();
    let solver = CpuCholesky;
    let load_case = &model.load_cases[0];

    let mut pipeline = AnalysisPipeline::new(&model);
    let result = pipeline.run(&solver, load_case).unwrap();

    let temp_dir = TempDir::new().unwrap();
    let csv_path = temp_dir.path().join("displacements.csv");

    result.export_displacements_csv(&csv_path).unwrap();

    assert!(csv_path.exists());
    let content = std::fs::read_to_string(&csv_path).unwrap();

    assert!(content.contains("NodeID"));
    assert!(content.contains("TX(m)"));
    assert!(content.contains("TY(m)"));
    assert!(content.contains("TZ(m)"));
    assert!(content.contains("RX(rad)"));
    assert!(content.contains("RY(rad)"));
    assert!(content.contains("RZ(rad)"));

    let lines: Vec<_> = content.lines().collect();
    assert_eq!(lines.len(), result.displacements.len() + 1);
}

#[test]
fn test_export_reactions_csv() {
    let model = create_simple_beam_model();
    let solver = CpuCholesky;
    let load_case = &model.load_cases[0];

    let mut pipeline = AnalysisPipeline::new(&model);
    let result = pipeline.run(&solver, load_case).unwrap();

    let temp_dir = TempDir::new().unwrap();
    let csv_path = temp_dir.path().join("reactions.csv");

    result.export_reactions_csv(&csv_path).unwrap();

    assert!(csv_path.exists());
    let content = std::fs::read_to_string(&csv_path).unwrap();

    assert!(content.contains("SupportID"));
    assert!(content.contains("NodeID"));
    assert!(content.contains("FX(N)"));
    assert!(content.contains("FY(N)"));
    assert!(content.contains("FZ(N)"));
    assert!(content.contains("MX(Nm)"));
    assert!(content.contains("MY(Nm)"));
    assert!(content.contains("MZ(Nm)"));

    let lines: Vec<_> = content.lines().collect();
    assert_eq!(lines.len(), result.reactions.len() + 1);
}

#[test]
fn test_export_beam_forces_csv() {
    let model = create_simple_beam_model();
    let solver = CpuCholesky;
    let load_case = &model.load_cases[0];

    let mut pipeline = AnalysisPipeline::new(&model);
    let result = pipeline.run(&solver, load_case).unwrap();

    let temp_dir = TempDir::new().unwrap();
    let csv_path = temp_dir.path().join("beam_forces.csv");

    result.export_beam_forces_csv(0, &csv_path).unwrap();

    assert!(csv_path.exists());
    let content = std::fs::read_to_string(&csv_path).unwrap();

    assert!(content.contains("Position(m)"));
    assert!(content.contains("Axial(N)"));
    assert!(content.contains("ShearY(N)"));
    assert!(content.contains("ShearZ(N)"));
    assert!(content.contains("MomentY(Nm)"));
    assert!(content.contains("MomentZ(Nm)"));
    assert!(content.contains("Torsion(Nm)"));
}

#[test]
fn test_csv_export_fails_for_nonexistent_element() {
    let model = create_simple_beam_model();
    let solver = CpuCholesky;
    let load_case = &model.load_cases[0];

    let mut pipeline = AnalysisPipeline::new(&model);
    let result = pipeline.run(&solver, load_case).unwrap();

    let temp_dir = TempDir::new().unwrap();
    let csv_path = temp_dir.path().join("beam_forces.csv");

    let export_result = result.export_beam_forces_csv(9999, &csv_path);
    assert!(export_result.is_err());
}

fn create_simple_beam_model() -> StructuralModel {
    let mut builder = ModelBuilder::new("Simple Beam Test");

    let concrete = materials::concrete_c30_37();
    builder.add_material(concrete);

    let section = sections::rectangular(0.3, 0.3);

    let span = 10.0;
    let num_elements = 10;

    let nodes: Vec<_> = (0..=num_elements)
        .map(|i| {
            let x = (i as f64) * span / (num_elements as f64);
            builder.add_node(Point3D { x, y: 0.0, z: 0.0 })
        })
        .collect();

    for i in 0..num_elements {
        builder
            .add_beam_element(nodes[i], nodes[i + 1], 0, section.clone())
            .unwrap();
    }

    builder.add_support(nodes[0], SupportType::Fixed).unwrap();
    builder
        .add_support(
            nodes[num_elements],
            SupportType::Roller {
                free_direction: Direction::X,
            },
        )
        .unwrap();

    builder
        .create_load_case("UDL", LoadType::Live)
        .add_uniform_load_on_all_elements(
            10000.0,
            Vector3D {
                x: 0.0,
                y: 0.0,
                z: -1.0,
            },
        )
        .unwrap()
        .finish();

    builder.build().unwrap()
}

#[test]
fn test_markdown_report_contains_code_citations() {
    let model = create_simple_beam_model();
    let solver = CpuCholesky;
    let load_case = &model.load_cases[0];

    let mut pipeline = AnalysisPipeline::new(&model);
    let result = pipeline.run(&solver, load_case).unwrap();

    let temp_dir = TempDir::new().unwrap();
    let md_path = temp_dir.path().join("report.md");

    result.export_markdown_report(&md_path, &model, load_case, "en").unwrap();

    assert!(md_path.exists());
    let content = std::fs::read_to_string(&md_path).unwrap();

    assert!(content.contains("EN 1992-1-1:2004"));
    assert!(content.contains("Table 3.1"));
    
    assert!(content.contains("Concrete C30/37"));
    
    assert!(content.contains("§ 7.4.1"));
    assert!(content.contains("L/250"));
}
