import { useState, useEffect } from 'react';
import { Brain } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from '@/components/ui/dialog';
import { Issue } from '@/types/dataset';
import { cn } from '@/lib/utils';

interface SmartFixDialogProps {
  issue: Issue | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSubmit: (response: string) => Promise<void>;
}

export const SmartFixDialog = ({ issue, open, onOpenChange, onSubmit }: SmartFixDialogProps) => {
  const [typedQuestion, setTypedQuestion] = useState('');
  const [showButtons, setShowButtons] = useState<number>(0);
  const [customAnswer, setCustomAnswer] = useState('');
  const [showCustomInput, setShowCustomInput] = useState(false);
  const [selectedOption, setSelectedOption] = useState<string | null>(null);

  const questions = {
    supplier_variations: 'Did something change in your business process, or is this data entry inconsistency?',
    discount_context: 'How should we interpret these discount values?',
    category_drift: 'Was this category change intentional or a data entry issue?',
  };

  const examples = {
    supplier_variations: 'Pharmax appears as: Pharmax, Pharmax Ltd, PHARMAX, Pharmax Ireland...',
    discount_context: 'Discount column shows values without context: 10, 15, 20...',
    category_drift: 'Categories changed over time: "Supplements" → "Health & Wellness"',
  };

  const question = issue ? questions[issue.type as keyof typeof questions] || '' : '';

  useEffect(() => {
    if (!open || !issue) {
      setTypedQuestion('');
      setShowButtons(0);
      setShowCustomInput(false);
      setSelectedOption(null);
      setCustomAnswer('');
      return;
    }

    // Reset state when dialog opens
    setTypedQuestion('');
    setShowButtons(0);

    // Typing animation
    let currentIndex = 0;
    const typingInterval = setInterval(() => {
      if (currentIndex <= question.length) {
        setTypedQuestion(question.slice(0, currentIndex));
        currentIndex++;
      } else {
        clearInterval(typingInterval);
        // Start showing buttons with stagger after typing completes
        setTimeout(() => {
          for (let i = 1; i <= 4; i++) {
            setTimeout(() => setShowButtons(i), i * 200);
          }
        }, 300);
      }
    }, 40);

    return () => clearInterval(typingInterval);
  }, [open, issue, question]);

  const handleOptionClick = (option: string) => {
    setSelectedOption(option);
    if (option === 'custom') {
      setShowCustomInput(true);
    } else {
      setShowCustomInput(false);
    }
  };

  const handleSubmit = async () => {
    if (selectedOption === 'custom' && customAnswer) {
      await onSubmit(customAnswer);
    } else if (selectedOption) {
      await onSubmit(selectedOption);
    }
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
              We found {issue.affectedRows?.toLocaleString()} {issue.description.toLowerCase()}
            </p>
            {issue.temporalPattern && (
              <p className="text-xs text-muted-foreground italic">{issue.temporalPattern}</p>
            )}
          </div>

          {/* Examples Section */}
          <div 
            className="p-4 rounded-lg border border-border animate-fade-in"
            style={{ 
              backgroundColor: 'hsl(var(--accent))',
              animationDelay: '200ms',
              animationFillMode: 'backwards'
            }}
          >
            <h4 className="font-semibold text-foreground mb-2">Examples Found</h4>
            <p className="text-sm text-muted-foreground">
              {examples[issue.type as keyof typeof examples]}
            </p>
          </div>

          {/* Typewriter Question */}
          <div className="min-h-[60px]">
            <p className="text-sm font-medium text-foreground">
              {typedQuestion}
              {typedQuestion.length < question.length && (
                <span className="animate-pulse">|</span>
              )}
            </p>
          </div>

          {/* Staggered Option Buttons */}
          <div className="grid grid-cols-2 gap-2">
            {options.map((option, index) => (
              <Button
                key={option.key}
                variant={selectedOption === option.key ? 'default' : 'outline'}
                className={cn(
                  'justify-start transition-all',
                  showButtons > index ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-2'
                )}
                style={{
                  transition: 'opacity 0.3s ease-out, transform 0.3s ease-out',
                }}
                onClick={() => handleOptionClick(option.key)}
                disabled={showButtons <= index}
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
        <div className="flex justify-end gap-2 mt-6 pt-4 border-t">
          <Button 
            variant="outline" 
            onClick={() => onOpenChange(false)}
          >
            Cancel
          </Button>
          <Button
            onClick={handleSubmit}
            disabled={!selectedOption || (selectedOption === 'custom' && !customAnswer.trim())}
            className={cn(
              'transition-all',
              showButtons === 4 ? 'opacity-100 scale-100' : 'opacity-0 scale-95'
            )}
          >
            Submit
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
};
