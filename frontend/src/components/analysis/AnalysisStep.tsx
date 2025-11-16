import { useState } from 'react';
import { Play, Loader2, CheckCircle2, XCircle, AlertTriangle, Wrench, Brain, ChevronRight } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { useDataset } from '@/context/DatasetContext';
import { apiClient } from '@/lib/apiClient';
import { useToast } from '@/hooks/use-toast';
import { StreamMessage, Issue } from '@/types/dataset';
import { cn } from '@/lib/utils';
import { SmartFixDialog } from './SmartFixDialog';

const severityConfig = {
  low: { icon: AlertTriangle, color: 'text-yellow-600', bg: 'bg-yellow-100 dark:bg-yellow-900' },
  medium: { icon: AlertTriangle, color: 'text-orange-600', bg: 'bg-orange-100 dark:bg-orange-900' },
  high: { icon: XCircle, color: 'text-red-600', bg: 'bg-red-100 dark:bg-red-900' },
};

export const AnalysisStep = () => {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [streamLogs, setStreamLogs] = useState<StreamMessage[]>([]);
  const [acceptedIssues, setAcceptedIssues] = useState<Set<string>>(new Set());
  const [openCategory, setOpenCategory] = useState<'quick_fixes' | 'smart_fixes' | null>(null);
  const [selectedIssue, setSelectedIssue] = useState<Issue | null>(null);
  const [smartFixResponse, setSmartFixResponse] = useState<string>('');
  const { datasetId, analysisResult, setAnalysisResult } = useDataset();
  const { toast } = useToast();

  const handleStartAnalysis = async () => {
    if (!datasetId) return;

    setIsAnalyzing(true);
    setStreamLogs([]);

    try {
      // Stream the analysis logs
      for await (const message of apiClient.streamAnalysis(datasetId)) {
        setStreamLogs(prev => [...prev, message]);
      }

      // Get the final result
      const result = await apiClient.analyzeDataset(datasetId);
      setAnalysisResult(result);
      
      toast({
        title: 'Analysis complete',
        description: `Found ${result.issues.length} data quality issues`,
      });
    } catch (error) {
      toast({
        title: 'Analysis failed',
        description: 'There was an error analyzing your dataset',
        variant: 'destructive',
      });
    } finally {
      setIsAnalyzing(false);
    }
  };

  const toggleIssue = (issueId: string) => {
    setAcceptedIssues(prev => {
      const newSet = new Set(prev);
      if (newSet.has(issueId)) {
        newSet.delete(issueId);
      } else {
        newSet.add(issueId);
      }
      return newSet;
    });
  };

  const handleApplyChanges = async () => {
    if (!datasetId || acceptedIssues.size === 0) return;

    try {
      const result = await apiClient.applyChanges(datasetId, Array.from(acceptedIssues));
      toast({
        title: 'Changes applied',
        description: result.message,
      });
    } catch (error) {
      toast({
        title: 'Failed to apply changes',
        description: 'There was an error applying the changes',
        variant: 'destructive',
      });
    }
  };

  const handleQuickFix = async (issueId: string, apply: boolean) => {
    if (apply) {
      try {
        await apiClient.applyChanges(datasetId!, [issueId]);
        toast({
          title: 'Fix applied',
          description: 'The issue has been resolved successfully',
        });
        setSelectedIssue(null);
      } catch (error) {
        toast({
          title: 'Failed to apply fix',
          description: 'There was an error applying the fix',
          variant: 'destructive',
        });
      }
    } else {
      setSelectedIssue(null);
    }
  };

  const handleSmartFixResponse = async (response: string) => {
    if (!datasetId || !selectedIssue) return;

    try {
      await apiClient.submitSmartFix(datasetId, selectedIssue.id, response);
      toast({
        title: 'Response recorded',
        description: 'Your answer has been captured for processing',
      });
    } catch (error) {
      toast({
        title: 'Failed to save response',
        description: 'There was an error submitting your answer',
        variant: 'destructive',
      });
    } finally {
      setSelectedIssue(null);
      setSmartFixResponse('');
    }
  };

  const renderIssue = (issue: Issue) => {
    const { icon: Icon, color, bg } = severityConfig[issue.severity];

    return (
      <Card 
        key={issue.id} 
        className="cursor-pointer hover:shadow-md transition-all"
        onClick={() => setSelectedIssue(issue)}
      >
        <CardContent className="p-4">
          <div className="flex items-start gap-3">
            <div className={cn('p-1.5 rounded-lg', bg)}>
              <Icon className={cn('w-4 h-4', color)} />
            </div>
            <div className="flex-1">
              <div className="flex items-center gap-2 mb-1">
                <h4 className="font-semibold text-foreground">{issue.description}</h4>
                <Badge variant="outline" className="text-xs">
                  {issue.severity}
                </Badge>
              </div>
              <p className="text-sm text-muted-foreground mb-2">
                {issue.affectedRows?.toLocaleString()} rows affected • {issue.affectedColumns.join(', ')}
              </p>
              {issue.temporalPattern && (
                <p className="text-xs text-muted-foreground italic">{issue.temporalPattern}</p>
              )}
            </div>
            <ChevronRight className="w-4 h-4 text-muted-foreground" />
          </div>
        </CardContent>
      </Card>
    );
  };

  return (
    <div className="max-w-5xl mx-auto">
      <div className="mb-8">
        <h2 className="text-3xl font-bold text-foreground mb-2">Analysis & Cleaning</h2>
        <p className="text-muted-foreground">
          Run AI-powered analysis to detect and fix data quality issues
        </p>
      </div>

      {!isAnalyzing && !analysisResult && (
        <Card className="mb-8">
          <CardHeader>
            <CardTitle>Ready to Analyze</CardTitle>
            <CardDescription>
              The AI will detect missing values, outliers, inconsistencies, and suggest cleaning operations
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Button size="lg" onClick={handleStartAnalysis}>
              <Play className="w-4 h-4 mr-2" />
              Run Analysis and Cleaning
            </Button>
          </CardContent>
        </Card>
      )}

      {(isAnalyzing || streamLogs.length > 0) && (
        <Card className="mb-8">
          <CardHeader>
            <div className="flex items-center gap-2">
              {isAnalyzing && <Loader2 className="w-5 h-5 animate-spin text-primary" />}
              <CardTitle>Analysis Log</CardTitle>
            </div>
          </CardHeader>
          <CardContent>
            <ScrollArea className="h-[300px] w-full rounded-lg border border-border bg-muted/30 p-4">
              <div className="space-y-2 font-mono text-sm">
                {streamLogs.map((log, index) => (
                  <div
                    key={index}
                    className={cn(
                      'flex items-start gap-2 py-1',
                      log.type === 'issue' && 'text-orange-600 dark:text-orange-400',
                      log.type === 'complete' && 'text-green-600 dark:text-green-400 font-semibold'
                    )}
                  >
                    <span className="text-muted-foreground text-xs mt-0.5">
                      [{new Date(log.timestamp).toLocaleTimeString()}]
                    </span>
                    <span>{log.message}</span>
                  </div>
                ))}
              </div>
            </ScrollArea>
          </CardContent>
        </Card>
      )}

      {analysisResult && analysisResult.issues.length > 0 && (
        <>
          <div className="mb-6">
            <h3 className="text-xl font-semibold text-foreground mb-2">Issue Categories</h3>
            <p className="text-sm text-muted-foreground mb-6">
              Review and select issues to address
            </p>

            <div className="space-y-4">
              {/* Quick Fixes Card */}
              {(() => {
                const quickFixes = analysisResult.issues.filter(i => i.category === 'quick_fixes');
                const totalRows = quickFixes.reduce((sum, issue) => sum + (issue.affectedRows || 0), 0);
                
                return (
                  <Card 
                    className="cursor-pointer hover:shadow-lg transition-all border-l-4 border-l-primary bg-gradient-to-r from-primary/5 to-transparent"
                    onClick={() => setOpenCategory('quick_fixes')}
                  >
                    <CardContent className="p-6">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-4">
                          <div className="p-3 rounded-lg bg-primary/10">
                            <Wrench className="w-6 h-6 text-primary" />
                          </div>
                          <div>
                            <h4 className="text-lg font-semibold text-foreground flex items-center gap-2">
                              Quick Fixes
                              <Badge variant="secondary">{quickFixes.length}</Badge>
                            </h4>
                            <p className="text-sm text-muted-foreground">
                              Technical data quality issues • {totalRows.toLocaleString()} rows affected
                            </p>
                          </div>
                        </div>
                        <ChevronRight className="w-5 h-5 text-muted-foreground" />
                      </div>
                    </CardContent>
                  </Card>
                );
              })()}

              {/* Smart Fixes Card */}
              {(() => {
                const smartFixes = analysisResult.issues.filter(i => i.category === 'smart_fixes');
                const totalRows = smartFixes.reduce((sum, issue) => sum + (issue.affectedRows || 0), 0);
                
                return (
                  <Card 
                    className="cursor-pointer hover:shadow-lg transition-all border-l-4 border-l-secondary bg-gradient-to-r from-secondary/5 to-transparent"
                    onClick={() => setOpenCategory('smart_fixes')}
                  >
                    <CardContent className="p-6">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-4">
                          <div className="p-3 rounded-lg bg-secondary/10">
                            <Brain className="w-6 h-6 text-secondary" />
                          </div>
                          <div>
                            <h4 className="text-lg font-semibold text-foreground flex items-center gap-2">
                              Smart Fixes
                              <Badge variant="secondary">{smartFixes.length}</Badge>
                            </h4>
                            <p className="text-sm text-muted-foreground">
                              Business logic issues requiring context • {totalRows.toLocaleString()} rows affected
                            </p>
                          </div>
                        </div>
                        <ChevronRight className="w-5 h-5 text-muted-foreground" />
                      </div>
                    </CardContent>
                  </Card>
                );
              })()}
            </div>
          </div>

          {/* Dialog for Quick Fixes */}
          <Dialog open={openCategory === 'quick_fixes'} onOpenChange={(open) => !open && setOpenCategory(null)}>
            <DialogContent className="max-w-3xl max-h-[80vh] overflow-y-auto">
              <DialogHeader>
                <DialogTitle className="flex items-center gap-2">
                  <Wrench className="w-5 h-5 text-primary" />
                  Quick Fixes - Technical Issues
                </DialogTitle>
                <DialogDescription>
                  Straightforward data quality fixes that can be applied automatically
                </DialogDescription>
              </DialogHeader>
              <div className="space-y-3 mt-4">
                {analysisResult.issues
                  .filter(i => i.category === 'quick_fixes')
                  .map(renderIssue)}
              </div>
              <div className="flex justify-end gap-2 mt-6 pt-4 border-t">
                <Button variant="outline" onClick={() => setOpenCategory(null)}>
                  Close
                </Button>
                <Button 
                  onClick={() => {
                    const quickFixIds = analysisResult.issues
                      .filter(i => i.category === 'quick_fixes')
                      .map(i => i.id);
                    quickFixIds.forEach(id => {
                      if (!acceptedIssues.has(id)) {
                        toggleIssue(id);
                      }
                    });
                    setOpenCategory(null);
                    handleApplyChanges();
                  }}
                  disabled={acceptedIssues.size === 0}
                >
                  <CheckCircle2 className="w-4 h-4 mr-2" />
                  Fix These
                </Button>
              </div>
            </DialogContent>
          </Dialog>

          {/* Dialog for Smart Fixes */}
          <Dialog open={openCategory === 'smart_fixes'} onOpenChange={(open) => !open && setOpenCategory(null)}>
            <DialogContent className="max-w-3xl max-h-[80vh] overflow-y-auto">
              <DialogHeader>
                <DialogTitle className="flex items-center gap-2">
                  <Brain className="w-5 h-5 text-secondary" />
                  Smart Fixes - Business Logic Issues
                </DialogTitle>
                <DialogDescription>
                  Issues that require business context and careful review before fixing
                </DialogDescription>
              </DialogHeader>
              <div className="space-y-3 mt-4">
                {analysisResult.issues
                  .filter(i => i.category === 'smart_fixes')
                  .map(renderIssue)}
              </div>
              <div className="flex justify-end gap-2 mt-6 pt-4 border-t">
                <Button variant="outline" onClick={() => setOpenCategory(null)}>
                  Close
                </Button>
              </div>
            </DialogContent>
          </Dialog>

          {/* Quick Fix Resolution Dialog */}
          <Dialog open={!!selectedIssue && selectedIssue.category === 'quick_fixes'} onOpenChange={(open) => !open && setSelectedIssue(null)}>
            <DialogContent className="max-w-2xl">
              {selectedIssue && selectedIssue.category === 'quick_fixes' && (
                <>
                  <DialogHeader>
                    <DialogTitle>Quick Fix Resolution</DialogTitle>
                    <DialogDescription>Review and apply this data quality fix</DialogDescription>
                  </DialogHeader>
                  <div className="space-y-4 mt-4">
                    <div className="p-4 bg-muted rounded-lg">
                      <h4 className="font-semibold text-foreground mb-2">Problem Detected</h4>
                      <p className="text-sm text-muted-foreground">
                        Found {selectedIssue.affectedRows?.toLocaleString()} rows with {selectedIssue.description.toLowerCase()} in {selectedIssue.affectedColumns.join(', ')}
                      </p>
                    </div>
                    
                    <div className="p-4 bg-primary/5 border border-primary/20 rounded-lg">
                      <h4 className="font-semibold text-foreground mb-3">Proposed Fix</h4>
                      <div className="space-y-2">
                        <div className="flex items-center gap-3 text-sm">
                          <span className="text-muted-foreground font-mono bg-muted px-2 py-1 rounded">Before:</span>
                          <span className="font-mono">" Vitamin D"</span>
                        </div>
                        <div className="flex items-center gap-3 text-sm">
                          <span className="text-primary font-mono bg-primary/10 px-2 py-1 rounded">After:</span>
                          <span className="font-mono">"Vitamin D"</span>
                        </div>
                      </div>
                      <p className="text-sm text-muted-foreground mt-3">{selectedIssue.suggestedAction}</p>
                    </div>
                  </div>
                  <div className="flex justify-end gap-2 mt-6">
                    <Button variant="outline" onClick={() => handleQuickFix(selectedIssue.id, false)}>
                      No, skip
                    </Button>
                    <Button onClick={() => handleQuickFix(selectedIssue.id, true)}>
                      <CheckCircle2 className="w-4 h-4 mr-2" />
                      Yes, fix it
                    </Button>
                  </div>
                </>
              )}
            </DialogContent>
          </Dialog>

          {/* Smart Fix Dialog with Animations */}
          <SmartFixDialog
            issue={selectedIssue?.category === 'smart_fixes' ? selectedIssue : null}
            open={!!selectedIssue && selectedIssue.category === 'smart_fixes'}
            onOpenChange={(open) => !open && setSelectedIssue(null)}
            onSubmit={handleSmartFixResponse}
          />
        </>
      )}

      {analysisResult && analysisResult.issues.length === 0 && (
        <Card>
          <CardContent className="py-12 text-center">
            <CheckCircle2 className="w-12 h-12 text-success mx-auto mb-4" />
            <h3 className="text-xl font-semibold text-foreground mb-2">All Clean!</h3>
            <p className="text-muted-foreground">No data quality issues detected</p>
          </CardContent>
        </Card>
      )}
    </div>
  );
};
