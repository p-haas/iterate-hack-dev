import { useState, useEffect } from 'react';
import { Brain, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from '@/components/ui/dialog';
import { Issue } from '@/types/dataset';

interface SmartFixDialogProps {
  issue: Issue | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSubmit: (response: string) => Promise<void>;
  onReject?: () => Promise<void> | void;
  decisionInProgress?: boolean;
}

export const SmartFixDialog = ({ issue, open, onOpenChange, onSubmit, onReject, decisionInProgress }: SmartFixDialogProps) => {
  const [customAnswer, setCustomAnswer] = useState('');
  const [showCustomInput, setShowCustomInput] = useState(false);
  const [selectedOption, setSelectedOption] = useState<string | null>(null);

  const formatInvestigationValue = (value: unknown) => {
    if (value === null || typeof value === 'undefined') {
      return 'No investigation output provided.';
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

  useEffect(() => {
    if (!open || !issue) {
      setShowCustomInput(false);
      setSelectedOption(null);
      setCustomAnswer('');
      return;
    }
  }, [open, issue]);

  const handleOptionClick = (option: string) => {
    setSelectedOption(option);
    if (option === 'custom') {
      setShowCustomInput(true);
    } else {
      setShowCustomInput(false);
    }
  };

  const handleSubmit = async () => {
    if (decisionInProgress) return;
    if (selectedOption === 'custom' && customAnswer) {
      await onSubmit(customAnswer);
    } else if (selectedOption) {
      await onSubmit(selectedOption);
    }
  };

  const handleReject = () => {
    if (decisionInProgress || !onReject) return;
    void onReject();
  };

  if (!issue) return null;

  const options = [
    { key: 'intentional', label: 'This is intentional - we changed processes' },
    { key: 'standardize', label: 'This is a mistake - standardize everything' },
    { key: 'keep', label: 'Keep it as is' },
    { key: 'custom', label: 'Custom answer' },
  ];

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Brain className="w-5 h-5 text-secondary" />
            Smart Fix Discussion
          </DialogTitle>
          <DialogDescription>
            Help us understand the context of this issue
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 mt-4">
          {/* Anomaly Detected Section */}
          <div className="p-4 bg-muted rounded-lg animate-fade-in">
            <h4 className="font-semibold text-foreground mb-2">Anomaly Detected</h4>
            <p className="text-sm text-muted-foreground mb-2">
              {issue.description}
            </p>
            <p className="text-xs text-muted-foreground">
              Columns: {issue.affectedColumns.join(', ')} • Severity: {issue.severity}
            </p>
            {issue.temporalPattern && (
              <p className="text-xs text-muted-foreground italic">{issue.temporalPattern}</p>
            )}
          </div>

          {/* Examples Section */}
          {issue.investigation && (
            <div className="space-y-4">
              {/* Evidence Examples */}
              {issue.investigation.evidence && (
                <div className="p-4 rounded-lg border border-primary/20 bg-primary/5">
                  <h4 className="font-semibold text-foreground mb-3 text-sm uppercase tracking-wide flex items-center gap-2">
                    <span className="text-primary">📋</span> Evidence-Based Examples
                  </h4>
                  {issue.investigation.evidence.pattern_description && (
                    <p className="text-sm text-muted-foreground mb-3 italic">
                      {issue.investigation.evidence.pattern_description}
                    </p>
                  )}
                  <div className="grid gap-4 md:grid-cols-2">
                    <div>
                      <p className="text-xs font-semibold text-destructive uppercase mb-2">❌ Current (Problematic)</p>
                      <ul className="space-y-1">
                        {issue.investigation.evidence.examples_current.slice(0, 3).map((example, idx) => (
                          <li key={idx} className="text-sm font-mono bg-destructive/10 px-2 py-1 rounded border border-destructive/20">
                            {example}
                          </li>
                        ))}
                      </ul>
                    </div>
                    <div>
                      <p className="text-xs font-semibold text-green-600 dark:text-green-400 uppercase mb-2">✓ Should Be (Fixed)</p>
                      <ul className="space-y-1">
                        {issue.investigation.evidence.examples_fixed.slice(0, 3).map((example, idx) => (
                          <li key={idx} className="text-sm font-mono bg-green-50 dark:bg-green-950 px-2 py-1 rounded border border-green-200 dark:border-green-800">
                            {example}
                          </li>
                        ))}
                      </ul>
                    </div>
                  </div>
                  {issue.investigation.evidence.fix_strategy && (
                    <div className="mt-3 p-3 rounded bg-muted/50 border border-border">
                      <p className="text-xs font-semibold text-muted-foreground uppercase mb-1">Fix Strategy</p>
                      <p className="text-sm text-foreground">{issue.investigation.evidence.fix_strategy}</p>
                    </div>
                  )}
                </div>
              )}
              
              {/* Technical Details */}
              <div className="grid gap-4 md:grid-cols-2">
                <div className="p-4 rounded-lg border border-border bg-muted/40">
                  <h4 className="font-semibold text-foreground mb-2 text-sm uppercase tracking-wide">Detector</h4>
                  <pre className="text-xs whitespace-pre-wrap font-mono text-foreground max-h-48 overflow-auto">
                    {issue.investigation.code || 'Not provided'}
                  </pre>
                </div>
                <div className="p-4 rounded-lg border border-border bg-muted/40">
                  <h4 className="font-semibold text-foreground mb-2 text-sm uppercase tracking-wide">Count</h4>
                  <pre className="text-xs whitespace-pre-wrap font-mono text-foreground max-h-48 overflow-auto">
                    {formatInvestigationValue(issue.investigation.output)}
                  </pre>
                </div>
              </div>
            </div>
          )}

          <div className="p-4 rounded-lg border border-dashed border-border">
            <h4 className="font-semibold text-foreground mb-1">Business Input Needed</h4>
            <p className="text-sm text-muted-foreground">
              {issue.suggestedAction || 'Explain how this should be handled.'}
            </p>
          </div>

          {/* Quick Reply Buttons */}
          <div className="grid grid-cols-2 gap-2">
            {options.map((option, index) => (
              <Button
                key={option.key}
                variant={selectedOption === option.key ? 'default' : 'outline'}
                className="justify-start transition-all"
                onClick={() => handleOptionClick(option.key)}
                disabled={!issue}
              >
                {option.label}
              </Button>
            ))}
          </div>

          {/* Custom Answer Input */}
          {showCustomInput && (
            <div className="space-y-2 animate-fade-in">
              <textarea
                className="w-full min-h-[100px] p-3 rounded-lg border border-border bg-background text-sm resize-none focus:outline-none focus:ring-2 focus:ring-primary"
                placeholder="Explain the context or provide specific instructions..."
                value={customAnswer}
                onChange={(e) => setCustomAnswer(e.target.value)}
              />
            </div>
          )}
        </div>

        {/* Submit Button */}
        <div className="flex justify-end gap-2 mt-6 pt-4 border-t flex-wrap">
          {onReject && (
            <Button
              type="button"
              variant="ghost"
              onClick={handleReject}
              disabled={decisionInProgress}
              className="mr-auto"
            >
              {decisionInProgress ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Saving...
                </>
              ) : (
                'Reject change'
              )}
            </Button>
          )}
          <Button
            variant="outline"
            onClick={() => onOpenChange(false)}
            disabled={decisionInProgress}
          >
            Cancel
          </Button>
          <Button
            onClick={handleSubmit}
            disabled={
              decisionInProgress ||
              !selectedOption ||
              (selectedOption === 'custom' && !customAnswer.trim())
            }
            className="transition-all"
          >
            {decisionInProgress ? (
              <>
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                Saving...
              </>
            ) : (
              'Submit'
            )}
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
};
