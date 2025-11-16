export interface Column {
  name: string;
  dataType: 'string' | 'numeric' | 'date' | 'categorical' | 'boolean';
  description: string;
  sampleValues?: string[];
}

export interface EvidenceExamples {
  examples_current: string[];
  examples_fixed: string[];
  pattern_description?: string;
  fix_strategy?: string;
}

export interface InvestigationDetails {
  code?: string;
  success?: boolean;
  output?: unknown;
  error?: string;
  execution_time_ms?: number;
  evidence?: EvidenceExamples;
}

export interface DatasetSummary {
  name: string;
  description: string;
  rowCount: number;
  columnCount: number;
  observations: string[];
}

export interface DatasetUnderstanding {
  columns: Column[];
  summary: DatasetSummary;
  suggested_context?: string;
}

export interface Issue {
  id: string;
  type: 'missing_values' | 'outliers' | 'inconsistent_categories' | 'invalid_dates' | 'duplicates' | 'whitespace' | 'near_duplicates' | 'supplier_variations' | 'discount_context' | 'category_drift';
  severity: 'low' | 'medium' | 'high';
  description: string;
  affectedColumns: string[];
  suggestedAction: string;
  accepted?: boolean;
  category: 'quick_fixes' | 'smart_fixes';
  affectedRows?: number;
  temporalPattern?: string;
  investigation?: InvestigationDetails;
}

export interface AnalysisResult {
  dataset_id?: string;
  issues: Issue[];
  summary: string;
  completedAt?: string;
}

export interface StreamMessage {
  type: 'log' | 'progress' | 'issue' | 'complete';
  message: string;
  timestamp: string;
  data?: any;
}
