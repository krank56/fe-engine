use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AuditEntry {
    pub timestamp: String,
    pub action: String,
    pub details: Vec<(String, AuditValue)>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AuditValue {
    String(String),
    Integer(i64),
    Float(f64),
    Boolean(bool),
    Array(Vec<AuditValue>),
}

impl AuditEntry {
    pub fn new(action: impl Into<String>) -> Self {
        Self {
            timestamp: chrono::Utc::now().to_rfc3339(),
            action: action.into(),
            details: Vec::new(),
        }
    }

    pub fn with_detail(mut self, key: impl Into<String>, value: AuditValue) -> Self {
        self.details.push((key.into(), value));
        self
    }

    pub fn with_details(mut self, details: Vec<(String, AuditValue)>) -> Self {
        self.details.extend(details);
        self
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct AuditTrail {
    pub entries: Vec<AuditEntry>,
}

impl AuditTrail {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    pub fn append(&mut self, entry: AuditEntry) {
        self.entries.push(entry);
    }

    pub fn append_action(&mut self, action: impl Into<String>) {
        self.entries.push(AuditEntry::new(action));
    }

    pub fn append_with_details(
        &mut self,
        action: impl Into<String>,
        details: Vec<(String, AuditValue)>,
    ) {
        self.entries
            .push(AuditEntry::new(action).with_details(details));
    }

    pub fn last_entry(&self) -> Option<&AuditEntry> {
        self.entries.last()
    }

    pub fn filter_by_action(&self, action: &str) -> Vec<&AuditEntry> {
        self.entries.iter().filter(|e| e.action == action).collect()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl From<i64> for AuditValue {
    fn from(v: i64) -> Self {
        AuditValue::Integer(v)
    }
}

impl From<f64> for AuditValue {
    fn from(v: f64) -> Self {
        AuditValue::Float(v)
    }
}

impl From<bool> for AuditValue {
    fn from(v: bool) -> Self {
        AuditValue::Boolean(v)
    }
}

impl From<String> for AuditValue {
    fn from(v: String) -> Self {
        AuditValue::String(v)
    }
}

impl From<&str> for AuditValue {
    fn from(v: &str) -> Self {
        AuditValue::String(v.to_string())
    }
}
