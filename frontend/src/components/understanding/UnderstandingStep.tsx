import { useEffect, useState } from 'react';
import { Loader2, Database, Edit2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { useDataset } from '@/context/DatasetContext';
import { apiClient } from '@/lib/apiClient';
import { useToast } from '@/hooks/use-toast';
import { Column } from '@/types/dataset';

const dataTypeColors: Record<Column['dataType'], string> = {
  string: 'bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-300',
  numeric: 'bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300',
  date: 'bg-purple-100 text-purple-700 dark:bg-purple-900 dark:text-purple-300',
  categorical: 'bg-orange-100 text-orange-700 dark:bg-orange-900 dark:text-orange-300',
  boolean: 'bg-pink-100 text-pink-700 dark:bg-pink-900 dark:text-pink-300',
};

export const UnderstandingStep = () => {
  const [isLoading, setIsLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const { datasetId, fileName, understanding, setUnderstanding, userContext, setUserContext, setCurrentStep, setAnalysisResult } = useDataset();
  const { toast } = useToast();

  useEffect(() => {
    if (datasetId && !understanding) {
      loadUnderstanding();
    } else {
      setIsLoading(false);
    }
  }, [datasetId]);

  const loadUnderstanding = async () => {
    if (!datasetId) return;

    setIsLoading(true);
    try {
      const [data, contextData] = await Promise.all([
        apiClient.getDatasetUnderstanding(datasetId),
        apiClient.getDatasetContext(datasetId),
      ]);

      setUnderstanding(data);
      if (contextData?.instructions) {
        setUserContext(contextData.instructions);
      }
    } catch (error) {
      toast({
        title: 'Error',
        description: 'Failed to load dataset understanding',
        variant: 'destructive',
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleSaveAndContinue = async () => {
    if (!datasetId) return;

    setIsSaving(true);
    try {
      await apiClient.saveContext(datasetId, userContext);
      setAnalysisResult(null);
      toast({
        title: 'Context saved',
        description: 'Your additional context has been saved',
      });
      setCurrentStep(2);
    } catch (error) {
      toast({
        title: 'Error',
        description: 'Failed to save context',
        variant: 'destructive',
      });
    } finally {
      setIsSaving(false);
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-20">
        <div className="text-center">
          <Loader2 className="w-12 h-12 animate-spin text-primary mx-auto mb-4" />
          <p className="text-lg font-medium text-foreground">Analyzing your dataset...</p>
          <p className="text-sm text-muted-foreground mt-2">This may take a moment</p>
        </div>
      </div>
    );
  }

  if (!understanding) {
    return (
      <div className="flex flex-col items-center justify-center py-20 text-center">
        <p className="text-lg font-medium text-destructive mb-2">Unable to load dataset understanding.</p>
        <p className="text-sm text-muted-foreground mb-4">Please try again or verify the dataset still exists on the server.</p>
        <Button variant="outline" onClick={loadUnderstanding}>Retry</Button>
      </div>
    );
  }

  return (
    <div className="max-w-5xl mx-auto">
      <div className="mb-8">
        <div className="flex items-center gap-3 mb-2">
          <Database className="w-6 h-6 text-primary" />
          <h2 className="text-3xl font-bold text-foreground">{fileName}</h2>
        </div>
        <p className="text-muted-foreground">Dataset Understanding</p>
      </div>

      {/* Global Summary */}
      <Card className="mb-8">
        <CardHeader>
          <CardTitle>Dataset Summary</CardTitle>
          <CardDescription>{understanding.summary.description}</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 gap-4 mb-4">
            <div className="bg-muted p-4 rounded-lg">
              <p className="text-sm text-muted-foreground">Rows</p>
              <p className="text-2xl font-bold text-foreground">{understanding.summary.rowCount.toLocaleString()}</p>
            </div>
            <div className="bg-muted p-4 rounded-lg">
              <p className="text-sm text-muted-foreground">Columns</p>
              <p className="text-2xl font-bold text-foreground">{understanding.summary.columnCount}</p>
            </div>
          </div>
          
          {understanding.summary.observations.length > 0 && (
            <div>
              <p className="text-sm font-medium text-foreground mb-2">Key Observations:</p>
              <ul className="space-y-1">
                {understanding.summary.observations.map((obs, i) => (
                  <li key={i} className="text-sm text-muted-foreground flex items-start">
                    <span className="text-primary mr-2">•</span>
                    {obs}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Columns */}
      <div className="mb-8">
        <h3 className="text-xl font-semibold text-foreground mb-4">Columns</h3>
        <div className="space-y-3">
          {understanding.columns.map((column) => (
            <Card key={column.name}>
              <CardContent className="p-4">
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-2">
                      <h4 className="font-mono font-semibold text-foreground">{column.name}</h4>
                      <Badge className={dataTypeColors[column.dataType]} variant="secondary">
                        {column.dataType}
                      </Badge>
                    </div>
                    <p className="text-sm text-muted-foreground mb-2">{column.description}</p>
                    {column.sampleValues && column.sampleValues.length > 0 && (
                      <div className="flex items-center gap-2 flex-wrap">
                        <span className="text-xs text-muted-foreground">Examples:</span>
                        {column.sampleValues.slice(0, 3).map((val, i) => (
                          <code key={i} className="text-xs bg-muted px-2 py-1 rounded">
                            {val || '(empty)'}
                          </code>
                        ))}
                      </div>
                    )}
                  </div>
                  <Button variant="ghost" size="sm">
                    <Edit2 className="w-4 h-4" />
                  </Button>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>

      {/* User Context */}
      <Card className="mb-8">
        <CardHeader>
          <CardTitle>Add Context or Instructions</CardTitle>
          <CardDescription>
            Help the AI better understand your data by providing additional context or specific instructions
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Textarea
            placeholder="Example: This dataset tracks monthly churn for our SaaS product. The customer_id should be treated as an identifier, not a numerical feature."
            value={userContext}
            onChange={(e) => setUserContext(e.target.value)}
            rows={4}
            className="resize-none"
          />
        </CardContent>
      </Card>

      <div className="flex justify-end">
        <Button
          size="lg"
          onClick={handleSaveAndContinue}
          disabled={isSaving}
        >
          {isSaving ? (
            <>
              <Loader2 className="w-4 h-4 mr-2 animate-spin" />
              Saving...
            </>
          ) : (
            'Save & Continue'
          )}
        </Button>
      </div>
    </div>
  );
};
