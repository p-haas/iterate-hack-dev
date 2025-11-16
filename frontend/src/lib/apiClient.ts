import { DatasetUnderstanding, AnalysisResult, StreamMessage, IssueDecisionResponse } from '@/types/dataset';

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

  async uploadDataset(file: File): Promise<{
    datasetId: string;
    fileName: string;
    fileType: string;
    fileSizeBytes: number;
    delimiter?: string | null;
    storagePath: string;
    uploadedAt: string;
  }> {
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

    const data = await response.json();
    return {
      datasetId: data.dataset_id,
      fileName: data.file_name,
      fileType: data.file_type,
      fileSizeBytes: data.file_size_bytes,
      delimiter: data.delimiter,
      storagePath: data.storage_path,
      uploadedAt: data.uploaded_at,
    };
  }

  async getDatasetUnderstanding(datasetId: string): Promise<DatasetUnderstanding> {
    return this.request<DatasetUnderstanding>(`/datasets/${datasetId}/understanding`);
  }

  async getDatasetContext(datasetId: string): Promise<{ instructions: string; column_edits?: any } | null> {
    try {
      return await this.request<{ instructions: string; column_edits?: any }>(`/datasets/${datasetId}/context`);
    } catch (error) {
      if (error instanceof APIError && error.status === 404) {
        return null;
      }
      throw error;
    }
  }

  async saveContext(datasetId: string, context: string, columnEdits?: any): Promise<void> {
    await this.request(`/datasets/${datasetId}/context`, {
      method: 'POST',
      body: JSON.stringify({ instructions: context, column_edits: columnEdits }),
    });
  }

  async analyzeDataset(datasetId: string): Promise<AnalysisResult> {
    return this.request<AnalysisResult>(`/datasets/${datasetId}/analysis`, {
      method: 'POST',
    });
  }

  async getAnalysisResult(datasetId: string): Promise<AnalysisResult | null> {
    try {
      return await this.request<AnalysisResult>(`/datasets/${datasetId}/analysis`);
    } catch (error) {
      if (error instanceof APIError && error.status === 404) {
        return null;
      }
      throw error;
    }
  }

  async* streamAnalysis(datasetId: string): AsyncGenerator<StreamMessage> {
    const response = await fetch(`${this.baseURL}/datasets/${datasetId}/analysis/stream`);

    if (!response.ok || !response.body) {
      throw new APIError('Failed to start analysis stream', response.status);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const parts = buffer.split('\n\n');
      buffer = parts.pop() || '';

      for (const part of parts) {
        const dataLine = part.split('\n').find(line => line.startsWith('data: '));
        if (!dataLine) continue;
        const payload = dataLine.replace('data: ', '');
        try {
          const message = JSON.parse(payload) as StreamMessage;
          yield message;
        } catch (error) {
          console.error('Failed to parse stream message', error, payload);
        }
      }
    }
  }

  async applyChanges(datasetId: string, acceptedIssueIds: string[]): Promise<{ applied: string[]; skipped: string[]; message: string }> {
    return this.request<{ applied: string[]; skipped: string[]; message: string }>(`/datasets/${datasetId}/apply`, {
      method: 'POST',
      body: JSON.stringify({ issueIds: acceptedIssueIds }),
    });
  }

  async submitSmartFix(datasetId: string, issueId: string, response: string): Promise<{ issue_id: string; response: string; updated_at: string }> {
    return this.request<{ issue_id: string; response: string; updated_at: string }>(`/datasets/${datasetId}/smart-fix`, {
      method: 'POST',
      body: JSON.stringify({ issueId, response }),
    });
  }

  async recordIssueDecision(datasetId: string, issueId: string, accepted: boolean, reason?: string): Promise<IssueDecisionResponse> {
    return this.request<IssueDecisionResponse>(`/datasets/${datasetId}/issues/decision`, {
      method: 'POST',
      body: JSON.stringify({ issueId, accepted, reason }),
    });
  }
}

export const apiClient = new APIClient();
