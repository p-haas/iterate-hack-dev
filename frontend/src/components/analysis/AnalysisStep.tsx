import { useEffect, useMemo, useState } from 'react';
import { Play, Loader2, CheckCircle2, XCircle, AlertTriangle, Wrench, Brain, ChevronRight, FileCode } from 'lucide-react';
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
  const [isFetchingAnalysis, setIsFetchingAnalysis] = useState(false);
  const [isApplyingQuickFixes, setIsApplyingQuickFixes] = useState(false);
  const [streamLogs, setStreamLogs] = useState<StreamMessage[]>([]);
  const [openCategory, setOpenCategory] = useState<'quick_fixes' | 'smart_fixes' | null>(null);
  const [selectedIssue, setSelectedIssue] = useState<Issue | null>(null);
  const [investigationIssue, setInvestigationIssue] = useState<Issue | null>(null);
  const { datasetId, analysisResult, setAnalysisResult } = useDataset();
  const { toast } = useToast();
  const quickFixes = useMemo(() => analysisResult?.issues.filter(i => i.category === 'quick_fixes') ?? [], [analysisResult]);
  const smartFixes = useMemo(() => analysisResult?.issues.filter(i => i.category === 'smart_fixes') ?? [], [analysisResult]);
  const quickFixRows = useMemo(() => quickFixes.reduce((sum, issue) => sum + (issue.affectedRows || 0), 0), [quickFixes]);
  const smartFixRows = useMemo(() => smartFixes.reduce((sum, issue) => sum + (issue.affectedRows || 0), 0), [smartFixes]);

  useEffect(() => {
    if (!datasetId || analysisResult || isAnalyzing) {
      return;
    }

    let cancelled = false;
    const fetchExistingAnalysis = async () => {
      setIsFetchingAnalysis(true);
      try {
        const existing = await apiClient.getAnalysisResult(datasetId);
        if (existing && !cancelled) {
          setAnalysisResult(existing);
        }
      } catch (error) {
        if (!cancelled) {
          toast({
            title: 'Unable to load previous analysis',
            description: 'Please launch a new analysis run to generate agent suggestions.',
            variant: 'destructive',
          });
        }
      } finally {
        if (!cancelled) {
          setIsFetchingAnalysis(false);
        }
      }
    };

    fetchExistingAnalysis();
    return () => {
      cancelled = true;
    };
  }, [datasetId, analysisResult, isAnalyzing, setAnalysisResult, toast]);

  const handleStartAnalysis = async () => {
    if (!datasetId) return;

    setAnalysisResult(null);
    setInvestigationIssue(null);
    setOpenCategory(null);
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

  const handleApplyAllQuickFixes = async () => {
    if (!datasetId || quickFixes.length === 0) return;

    setIsApplyingQuickFixes(true);
    try {
      const result = await apiClient.applyChanges(
        datasetId,
        quickFixes.map(issue => issue.id),
      );
      toast({
        title: 'Quick fixes applied',
        description: result.message,
      });
      setOpenCategory(null);
    } catch (error) {
      toast({
        title: 'Failed to apply fixes',
        description: 'There was an error applying the quick fixes',
        variant: 'destructive',
      });
    } finally {
      setIsApplyingQuickFixes(false);
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
    }
  };

  const formatInvestigationValue = (value: unknown) => {
    if (value === null || typeof value === 'undefined') {
      return 'No output captured';
    }
    if (typeof value === 'string') {
      return value;
    }
    try {
      return JSON.stringify(value, null, 2);
    } catch {
      return String(value);
    }
  };

  const renderIssue = (issue: Issue) => {
    const { icon: Icon, color, bg } = severityConfig[issue.severity];
    const affectedRowsText =
      typeof issue.affectedRows === 'number'
        ? `${issue.affectedRows.toLocaleString()} rows affected`
        : 'Impact pending';

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
                {affectedRowsText} • {issue.affectedColumns.join(', ')}
              </p>
              {issue.temporalPattern && (
                <p className="text-xs text-muted-foreground italic">{issue.temporalPattern}</p>
              )}
              {issue.investigation && (
                <Button
                  variant="ghost"
                  size="sm"
                  className="mt-2 px-2 h-8 text-xs"
                  onClick={(event) => {
                    event.stopPropagation();
                    setInvestigationIssue(issue);
                  }}
                >
                  <FileCode className="w-4 h-4 mr-1" />
                  View evidence
                </Button>
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

      {isFetchingAnalysis && !analysisResult && (
        <Card className="mb-8">
          <CardHeader>
            <CardTitle>Loading previous analysis</CardTitle>
            <CardDescription>The AI agent is fetching the latest quick + smart fixes for this dataset</CardDescription>
          </CardHeader>
          <CardContent className="flex items-center gap-3 text-muted-foreground">
            <Loader2 className="w-4 h-4 animate-spin" />
            <span>Checking cached results...</span>
          </CardContent>
        </Card>
      )}

      {!isAnalyzing && !analysisResult && !isFetchingAnalysis && (
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

      {analysisResult && (
        <Card className="mb-8">
          <CardHeader>
            <CardTitle>AI Agent Summary</CardTitle>
            {analysisResult.completedAt && (
              <CardDescription>
                Completed {new Date(analysisResult.completedAt).toLocaleString()}
              </CardDescription>
            )}
          </CardHeader>
          <CardContent>
            <p className="text-muted-foreground">{analysisResult.summary}</p>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mt-4">
              <div className="p-4 rounded-lg bg-muted">
                <p className="text-xs text-muted-foreground uppercase tracking-wide">Quick fixes</p>
                <p className="text-2xl font-semibold text-foreground">{quickFixes.length}</p>
                <p className="text-xs text-muted-foreground">
                  {quickFixRows.toLocaleString()} rows impacted
                </p>
              </div>
              <div className="p-4 rounded-lg bg-muted">
                <p className="text-xs text-muted-foreground uppercase tracking-wide">Smart fixes</p>
                <p className="text-2xl font-semibold text-foreground">{smartFixes.length}</p>
                <p className="text-xs text-muted-foreground">
                  {smartFixRows.toLocaleString()} rows impacted
                </p>
              </div>
            </div>
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
                          Technical data quality issues • {quickFixRows.toLocaleString()} rows affected
                        </p>
                      </div>
                    </div>
                    <ChevronRight className="w-5 h-5 text-muted-foreground" />
                  </div>
                </CardContent>
              </Card>

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
                          Business logic issues requiring context • {smartFixRows.toLocaleString()} rows affected
                        </p>
                      </div>
                    </div>
                    <ChevronRight className="w-5 h-5 text-muted-foreground" />
                  </div>
                </CardContent>
              </Card>
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
                {quickFixes.map(renderIssue)}
              </div>
              <div className="flex justify-end gap-2 mt-6 pt-4 border-t">
                <Button variant="outline" onClick={() => setOpenCategory(null)}>
                  Close
                </Button>
                <Button 
                  onClick={handleApplyAllQuickFixes}
                  disabled={quickFixes.length === 0 || isApplyingQuickFixes}
                >
                  {isApplyingQuickFixes ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      Applying...
                    </>
                  ) : (
                    <>
                      <CheckCircle2 className="w-4 h-4 mr-2" />
                      Fix These
                    </>
                  )}
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
                {smartFixes.map(renderIssue)}
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
                      <p className="text-sm text-muted-foreground mb-2">
                        {selectedIssue.description}
                      </p>
                      <p className="text-xs text-muted-foreground">
                        Columns: {selectedIssue.affectedColumns.join(', ')} • Rows affected: {selectedIssue.affectedRows?.toLocaleString() || 'n/a'}
                      </p>
                    </div>

                    <div className="p-4 bg-primary/5 border border-primary/20 rounded-lg">
                      <h4 className="font-semibold text-foreground mb-2">Suggested Action</h4>
                      <p className="text-sm text-muted-foreground">{selectedIssue.suggestedAction}</p>
                    </div>

                    {selectedIssue.investigation && (
                      <div className="space-y-4">
                        {/* Evidence Examples - Show first if available */}
                        {selectedIssue.investigation.evidence && (
                          <div className="p-4 rounded-lg border border-primary/20 bg-primary/5">
                            <h4 className="font-semibold text-foreground mb-3 text-sm uppercase tracking-wide flex items-center gap-2">
                              <span className="text-primary">📋</span> Evidence-Based Examples
                            </h4>
                            {selectedIssue.investigation.evidence.pattern_description && (
                              <p className="text-sm text-muted-foreground mb-3 italic">
                                {selectedIssue.investigation.evidence.pattern_description}
                              </p>
                            )}
                            <div className="grid gap-4 md:grid-cols-2">
                              <div>
                                <p className="text-xs font-semibold text-destructive uppercase mb-2">❌ Current (Problematic)</p>
                                <ul className="space-y-1">
                                  {selectedIssue.investigation.evidence.examples_current.slice(0, 5).map((example, idx) => (
                                    <li key={idx} className="text-sm font-mono bg-destructive/10 px-2 py-1 rounded border border-destructive/20">
                                      {example}
                                    </li>
                                  ))}
                                </ul>
                              </div>
                              <div>
                                <p className="text-xs font-semibold text-green-600 dark:text-green-400 uppercase mb-2">✓ Should Be (Fixed)</p>
                                <ul className="space-y-1">
                                  {selectedIssue.investigation.evidence.examples_fixed.slice(0, 5).map((example, idx) => (
                                    <li key={idx} className="text-sm font-mono bg-green-50 dark:bg-green-950 px-2 py-1 rounded border border-green-200 dark:border-green-800">
                                      {example}
                                    </li>
                                  ))}
                                </ul>
                              </div>
                            </div>
                            {selectedIssue.investigation.evidence.fix_strategy && (
                              <div className="mt-3 p-3 rounded bg-muted/50 border border-border">
                                <p className="text-xs font-semibold text-muted-foreground uppercase mb-1">Fix Strategy</p>
                                <p className="text-sm text-foreground">{selectedIssue.investigation.evidence.fix_strategy}</p>
                              </div>
                            )}
                          </div>
                        )}
                        
                        {/* Technical Details */}
                        <div className="grid gap-4 md:grid-cols-2">
                          <div className="p-4 rounded-lg border border-border bg-muted/40">
                            <h4 className="font-semibold text-foreground mb-2 text-sm uppercase tracking-wide">Detector</h4>
                            <pre className="text-xs whitespace-pre-wrap font-mono text-foreground max-h-48 overflow-auto">
                              {selectedIssue.investigation.code || 'Not provided'}
                            </pre>
                          </div>
                          <div className="p-4 rounded-lg border border-border bg-muted/40">
                            <h4 className="font-semibold text-foreground mb-2 text-sm uppercase tracking-wide">Count</h4>
                            <pre className="text-xs whitespace-pre-wrap font-mono text-foreground max-h-48 overflow-auto">
                              {formatInvestigationValue(selectedIssue.investigation.output)}
                            </pre>
                          </div>
                        </div>
                      </div>
                    )}
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

          {/* Investigation Evidence Dialog */}
          <Dialog open={!!investigationIssue} onOpenChange={(open) => !open && setInvestigationIssue(null)}>
            <DialogContent className="max-w-3xl">
              {investigationIssue?.investigation ? (
                <>
                  <DialogHeader>
                    <DialogTitle className="flex items-center gap-2">
                      <FileCode className="w-5 h-5 text-primary" />
                      Investigation Evidence
                    </DialogTitle>
                    <DialogDescription>
                      Code executed by the AI agent to diagnose <span className="font-semibold">{investigationIssue.description}</span>
                    </DialogDescription>
                  </DialogHeader>
                  <div className="space-y-4 mt-4">
                    <div className="flex items-center gap-3">
                      <Badge variant={investigationIssue.investigation.success ? 'secondary' : 'destructive'}>
                        {investigationIssue.investigation.success ? 'Execution succeeded' : 'Execution failed'}
                      </Badge>
                      {typeof investigationIssue.investigation.execution_time_ms === 'number' && (
                        <p className="text-xs text-muted-foreground">
                          {investigationIssue.investigation.execution_time_ms.toFixed(0)} ms runtime
                        </p>
                      )}
                    </div>
                    
                    {/* Evidence Examples Section */}
                    {investigationIssue.investigation.evidence && (
                      <div className="p-4 rounded-lg border border-primary/30 bg-primary/5">
                        <h4 className="font-semibold text-foreground mb-3 text-sm uppercase tracking-wide">
                          📋 Evidence-Based Examples
                        </h4>
                        {investigationIssue.investigation.evidence.pattern_description && (
                          <p className="text-sm text-muted-foreground mb-3 italic">
                            {investigationIssue.investigation.evidence.pattern_description}
                          </p>
                        )}
                        <div className="grid gap-4 md:grid-cols-2">
                          <div>
                            <p className="text-xs font-semibold text-destructive uppercase mb-2">❌ Current (Problematic)</p>
                            <ul className="space-y-1.5">
                              {investigationIssue.investigation.evidence.examples_current.map((example, idx) => (
                                <li key={idx} className="text-sm font-mono bg-destructive/10 px-3 py-1.5 rounded border border-destructive/20">
                                  {example}
                                </li>
                              ))}
                            </ul>
                          </div>
                          <div>
                            <p className="text-xs font-semibold text-green-600 dark:text-green-400 uppercase mb-2">✓ Should Be (Fixed)</p>
                            <ul className="space-y-1.5">
                              {investigationIssue.investigation.evidence.examples_fixed.map((example, idx) => (
                                <li key={idx} className="text-sm font-mono bg-green-50 dark:bg-green-950 px-3 py-1.5 rounded border border-green-200 dark:border-green-800 text-green-900 dark:text-green-100">
                                  {example}
                                </li>
                              ))}
                            </ul>
                          </div>
                        </div>
                        {investigationIssue.investigation.evidence.fix_strategy && (
                          <div className="mt-3 p-3 rounded bg-muted/50 border border-border">
                            <p className="text-xs font-semibold text-muted-foreground uppercase mb-1">Fix Strategy</p>
                            <p className="text-sm text-foreground">{investigationIssue.investigation.evidence.fix_strategy}</p>
                          </div>
                        )}
                      </div>
                    )}
                    
                    <div className="grid gap-4 md:grid-cols-2">
                      <div className="p-4 rounded-lg border border-border bg-muted/30">
                        <p className="text-xs font-semibold text-muted-foreground uppercase mb-2">Python code</p>
                        <pre className="text-xs whitespace-pre-wrap font-mono text-foreground max-h-60 overflow-auto">
                          {investigationIssue.investigation.code || 'Code snippet unavailable'}
                        </pre>
                      </div>
                      <div className="p-4 rounded-lg border border-border bg-muted/30">
                        <p className="text-xs font-semibold text-muted-foreground uppercase mb-2">Output</p>
                        <pre className="text-xs whitespace-pre-wrap font-mono text-foreground max-h-60 overflow-auto">
                          {formatInvestigationValue(investigationIssue.investigation.output)}
                        </pre>
                      </div>
                    </div>
                    {investigationIssue.investigation.error && (
                      <div className="p-3 rounded-lg bg-red-50 dark:bg-red-950 border border-red-200 dark:border-red-800 text-sm text-red-700 dark:text-red-200">
                        {investigationIssue.investigation.error}
                      </div>
                    )}
                  </div>
                </>
              ) : (
                <div className="py-6 text-sm text-muted-foreground text-center">
                  Investigation metadata is unavailable for this issue.
                </div>
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
