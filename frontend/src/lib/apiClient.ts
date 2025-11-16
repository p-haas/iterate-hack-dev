import { DatasetUnderstanding, AnalysisResult, StreamMessage } from '@/types/dataset';

const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));
const DEFAULT_BASE_URL = (import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000').replace(/\/$/, '');

class APIError extends Error {
  status: number;
  details?: unknown;

  constructor(message: string, status: number, details?: unknown) {
    super(message);
    this.name = 'APIError';
    this.status = status;
    this.details = details;
  }
}

class APIClient {
  private baseURL: string;

  constructor(baseURL: string = DEFAULT_BASE_URL) {
    this.baseURL = baseURL;
  }

  private async request<T>(path: string, options: RequestInit = {}): Promise<T> {
    const response = await fetch(`${this.baseURL}${path}`, {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    });

    if (!response.ok) {
      let body: any;
      try {
        body = await response.json();
      } catch {
        body = await response.text();
      }
      const message = typeof body === 'string' && body ? body : body?.detail || response.statusText;
      throw new APIError(message || 'Request failed', response.status, body);
    }

    if (response.status === 204) {
      return undefined as T;
    }

    return response.json() as Promise<T>;
  }

  async health(): Promise<{ status: string }> {
    return this.request<{ status: string }>('/health');
  }

  async uploadDataset(file: File): Promise<{ datasetId: string }> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${this.baseURL}/datasets`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new APIError(errorText || 'Failed to upload dataset', response.status);
    }

    return response.json();
  }

  async getDatasetUnderstanding(datasetId: string): Promise<DatasetUnderstanding> {
    return this.request<DatasetUnderstanding>(`/datasets/${datasetId}/understanding`);
  }

  async saveContext(datasetId: string, context: string, columnEdits?: any): Promise<void> {
    await delay(800);
    console.log('Saving context for', datasetId, ':', context, columnEdits);
  }

  async analyzeDataset(datasetId: string): Promise<AnalysisResult> {
    await delay(3000);
    return {
      issues: [
        // Quick Fixes - Technical data quality issues
        {
          id: 'issue_1',
          type: 'missing_values',
          severity: 'medium',
          description: 'Missing values in multiple columns',
          affectedColumns: ['churn_date', 'discount_amount'],
          suggestedAction: 'Fill or remove missing values based on business logic',
          accepted: false,
          category: 'quick_fixes',
          affectedRows: 342
        },
        {
          id: 'issue_2',
          type: 'whitespace',
          severity: 'low',
          description: 'Whitespace issues in text fields',
          affectedColumns: ['supplier_name', 'product_category'],
          suggestedAction: 'Trim leading and trailing whitespace',
          accepted: false,
          category: 'quick_fixes',
          affectedRows: 89
        },
        {
          id: 'issue_3',
          type: 'duplicates',
          severity: 'high',
          description: 'Duplicate rows detected',
          affectedColumns: ['customer_id', 'order_date'],
          suggestedAction: 'Remove exact duplicate entries',
          accepted: false,
          category: 'quick_fixes',
          affectedRows: 23
        },
        {
          id: 'issue_4',
          type: 'near_duplicates',
          severity: 'medium',
          description: 'Near-duplicate rows with minor differences',
          affectedColumns: ['supplier_name', 'product_name'],
          suggestedAction: 'Review and merge similar entries',
          accepted: false,
          category: 'quick_fixes',
          affectedRows: 156
        },
        // Smart Fixes - Business logic issues
        {
          id: 'issue_5',
          type: 'supplier_variations',
          severity: 'high',
          description: 'Supplier name variations',
          affectedColumns: ['supplier_name'],
          suggestedAction: 'Standardize supplier names across variations (e.g., "ABC Corp" vs "ABC Corporation")',
          accepted: false,
          category: 'smart_fixes',
          affectedRows: 234,
          temporalPattern: 'Starting April 2024'
        },
        {
          id: 'issue_6',
          type: 'discount_context',
          severity: 'medium',
          description: 'Discount context loss',
          affectedColumns: ['discount_amount', 'promotion_code'],
          suggestedAction: 'Link discounts to promotional campaigns for better tracking',
          accepted: false,
          category: 'smart_fixes',
          affectedRows: 421,
          temporalPattern: 'Q2 2024 promotions'
        },
        {
          id: 'issue_7',
          type: 'category_drift',
          severity: 'medium',
          description: 'Category drift over time',
          affectedColumns: ['product_category'],
          suggestedAction: 'Reconcile category changes and establish consistent taxonomy',
          accepted: false,
          category: 'smart_fixes',
          affectedRows: 187,
          temporalPattern: 'Starting April 2024'
        }
      ],
      summary: 'Analysis complete. Found 7 data quality issues: 4 quick fixes and 3 smart fixes.',
      completedAt: new Date().toISOString()
    };
  }

  async* streamAnalysis(datasetId: string): AsyncGenerator<StreamMessage> {
    const messages: StreamMessage[] = [
      {
        type: 'log',
        message: 'Starting dataset analysis...',
        timestamp: new Date().toISOString()
      },
      {
        type: 'progress',
        message: 'Loading dataset into memory',
        timestamp: new Date().toISOString()
      },
      {
        type: 'progress',
        message: 'Analyzing column types and distributions',
        timestamp: new Date().toISOString()
      },
      {
        type: 'log',
        message: 'Detecting missing values...',
        timestamp: new Date().toISOString()
      },
      {
        type: 'issue',
        message: 'Found 342 missing values in churn_date column',
        timestamp: new Date().toISOString()
      },
      {
        type: 'log',
        message: 'Checking categorical consistency...',
        timestamp: new Date().toISOString()
      },
      {
        type: 'issue',
        message: 'Detected inconsistent capitalization in plan_type',
        timestamp: new Date().toISOString()
      },
      {
        type: 'issue',
        message: 'Found mixed boolean formats in is_active',
        timestamp: new Date().toISOString()
      },
      {
        type: 'log',
        message: 'Analyzing numerical distributions...',
        timestamp: new Date().toISOString()
      },
      {
        type: 'issue',
        message: 'Identified 3 potential outliers in monthly_revenue',
        timestamp: new Date().toISOString()
      },
      {
        type: 'progress',
        message: 'Generating cleaning recommendations',
        timestamp: new Date().toISOString()
      },
      {
        type: 'complete',
        message: 'Analysis complete',
        timestamp: new Date().toISOString()
      }
    ];

    for (const msg of messages) {
      await delay(600 + Math.random() * 400);
      yield msg;
    }
  }

  async applyChanges(datasetId: string, acceptedIssueIds: string[]): Promise<void> {
    await delay(1000);
    console.log('Applying changes for issues:', acceptedIssueIds);
  }
}

export const apiClient = new APIClient();
