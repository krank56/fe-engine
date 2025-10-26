pub mod analysis;
pub mod audit;
pub mod builder;
pub mod export;
pub mod structure;
pub mod validation;

pub mod prelude {
    pub use crate::analysis::{AnalysisPipeline, CpuCholesky, LinearSolver};
    pub use crate::audit::AuditLogger;
    pub use crate::builder::materials;
    pub use crate::builder::sections;
    pub use crate::builder::ModelBuilder;
    pub use crate::structure::element::{ElementType, Plane};
    pub use crate::structure::geometry::{Point3D, Vector3D};
    pub use crate::structure::load::{LoadDistribution, LoadType};
    pub use crate::structure::material::{
        ConcreteGrade, ConcreteStandard, Material, MaterialType, SteelGrade, SteelStandard,
    };
    pub use crate::structure::model::{StructuralModel, UnitSystem};
    pub use crate::structure::section::Section;
    pub use crate::structure::support::{Direction, SupportType};
    pub use crate::validation::{BuildError, ValidationError};
}
