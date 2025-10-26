use super::trail::{AuditEntry, AuditTrail, AuditValue};
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone)]
pub struct AuditLogger {
    trail: Arc<Mutex<AuditTrail>>,
}

impl AuditLogger {
    pub fn new() -> Self {
        Self {
            trail: Arc::new(Mutex::new(AuditTrail::new())),
        }
    }

    pub fn log(&self, action: impl Into<String>) {
        if let Ok(mut trail) = self.trail.lock() {
            trail.append_action(action);
        }
    }

    pub fn log_with_details(&self, action: impl Into<String>, details: Vec<(String, AuditValue)>) {
        if let Ok(mut trail) = self.trail.lock() {
            trail.append_with_details(action, details);
        }
    }

    pub fn log_entry(&self, entry: AuditEntry) {
        if let Ok(mut trail) = self.trail.lock() {
            trail.append(entry);
        }
    }

    pub fn get_trail(&self) -> Option<AuditTrail> {
        self.trail.lock().ok().map(|t| t.clone())
    }

    pub fn into_trail(self) -> AuditTrail {
        Arc::try_unwrap(self.trail)
            .map(|mutex| mutex.into_inner().unwrap())
            .unwrap_or_else(|arc| arc.lock().unwrap().clone())
    }
}

impl Default for AuditLogger {
    fn default() -> Self {
        Self::new()
    }
}
