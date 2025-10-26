use fe_engine::prelude::*;
use fe_engine::audit::trail::{AuditTrail, AuditEntry, AuditValue};

#[test]
fn test_audit_trail_records_assembly_phase() {
    let model = create_simple_beam_model();
    let solver = CpuCholesky;
    let load_case = &model.load_cases[0];
    
    let mut pipeline = AnalysisPipeline::new(&model);
    let result = pipeline.run(&solver, load_case).unwrap();
    
    let audit = &pipeline.audit_trail;
    
    let assembly_entries = audit.filter_by_action("assemble_global_stiffness");
    assert!(!assembly_entries.is_empty(), "Should record stiffness matrix assembly");
    
    let load_entries = audit.filter_by_action("assemble_load_vector");
    assert!(!load_entries.is_empty(), "Should record load vector assembly");
}

#[test]
fn test_audit_trail_records_solver_details() {
    let model = create_simple_beam_model();
    let solver = CpuCholesky;
    let load_case = &model.load_cases[0];
    
    let mut pipeline = AnalysisPipeline::new(&model);
    let result = pipeline.run(&solver, load_case).unwrap();
    
    let audit = &pipeline.audit_trail;
    
    let solve_entries = audit.filter_by_action("solve_linear_system");
    assert!(!solve_entries.is_empty(), "Should record linear system solve");
    
    if let Some(entry) = solve_entries.first() {
        let has_matrix_size = entry.details.iter().any(|(k, _)| k.contains("matrix_size") || k.contains("dofs"));
        assert!(has_matrix_size, "Should record matrix dimensions");
    }
}

#[test]
fn test_audit_trail_records_boundary_conditions() {
    let model = create_simple_beam_model();
    let solver = CpuCholesky;
    let load_case = &model.load_cases[0];
    
    let mut pipeline = AnalysisPipeline::new(&model);
    let result = pipeline.run(&solver, load_case).unwrap();
    
    let audit = &pipeline.audit_trail;
    
    let bc_entries = audit.filter_by_action("apply_boundary_conditions");
    assert!(!bc_entries.is_empty(), "Should record boundary condition application");
}

#[test]
fn test_audit_trail_includes_element_force_recovery() {
    let model = create_simple_beam_model();
    let solver = CpuCholesky;
    let load_case = &model.load_cases[0];
    
    let mut pipeline = AnalysisPipeline::new(&model);
    let result = pipeline.run(&solver, load_case).unwrap();
    
    let audit = &pipeline.audit_trail;
    
    let force_entries = audit.filter_by_action("compute_element_forces");
    assert!(!force_entries.is_empty(), "Should record element force computation");
}

#[test]
fn test_audit_trail_includes_reaction_calculation() {
    let model = create_simple_beam_model();
    let solver = CpuCholesky;
    let load_case = &model.load_cases[0];
    
    let mut pipeline = AnalysisPipeline::new(&model);
    let result = pipeline.run(&solver, load_case).unwrap();
    
    let audit = &pipeline.audit_trail;
    
    let reaction_entries = audit.filter_by_action("compute_support_reactions");
    assert!(!reaction_entries.is_empty(), "Should record support reaction computation");
}

#[test]
fn test_audit_trail_is_chronological() {
    let model = create_simple_beam_model();
    let solver = CpuCholesky;
    let load_case = &model.load_cases[0];
    
    let mut pipeline = AnalysisPipeline::new(&model);
    let result = pipeline.run(&solver, load_case).unwrap();
    
    let audit = &pipeline.audit_trail;
    
    assert!(audit.len() >= 3, "Should have multiple audit entries");
    
    for i in 1..audit.entries.len() {
        let prev_time = chrono::DateTime::parse_from_rfc3339(&audit.entries[i - 1].timestamp).unwrap();
        let curr_time = chrono::DateTime::parse_from_rfc3339(&audit.entries[i].timestamp).unwrap();
        assert!(curr_time >= prev_time, "Audit entries should be chronological");
    }
}

#[test]
fn test_audit_trail_serializes_to_json() {
    let mut trail = AuditTrail::new();
    
    trail.append(
        AuditEntry::new("test_action")
            .with_detail("num_dofs", AuditValue::Integer(100))
            .with_detail("max_displacement", AuditValue::Float(0.012))
            .with_detail("converged", AuditValue::Boolean(true))
    );
    
    let json = serde_json::to_string(&trail).unwrap();
    assert!(json.contains("test_action"));
    assert!(json.contains("num_dofs"));
    
    let deserialized: AuditTrail = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.len(), 1);
    assert_eq!(deserialized.entries[0].action, "test_action");
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
