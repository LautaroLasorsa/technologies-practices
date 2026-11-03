import { useState } from 'react';
import type { UseFormSetError } from 'react-hook-form';
import type {
  ShiftWizardInput,
  ServerValidationErrorResponse,
  ShiftSubmitSuccessResponse,
} from '../../shared/schemas';

interface UseWizardSubmitReturn {
  isSubmitting: boolean;
  submitError: string | null;
  submitSuccess: ShiftSubmitSuccessResponse | null;
  submitWizard: (data: ShiftWizardInput) => Promise<boolean>;
  resetSubmitState: () => void;
}

export function useWizardSubmit(
  setError: UseFormSetError<ShiftWizardInput>
): UseWizardSubmitReturn {
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [submitSuccess, setSubmitSuccess] = useState<ShiftSubmitSuccessResponse | null>(null);

  const resetSubmitState = () => {
    setSubmitError(null);
    setSubmitSuccess(null);
  };

  const submitWizard = async (data: ShiftWizardInput): Promise<boolean> => {
    setIsSubmitting(true);
    setSubmitError(null);
    setSubmitSuccess(null);

    try {
      const response = await fetch('/api/shifts', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
      });

      // ----------------------------------------------------------------------
      // TODO(human) [Exercise 3: Client Server-Error Hydration]
      // ----------------------------------------------------------------------
      // Learning Objectives:
      // 1. Inspect HTTP response status (400 or 422):
      //    When the backend rejects the submission due to schema violations (400)
      //    or stateful database collisions (422), parse the JSON payload:
      //    const errData: ServerValidationErrorResponse = await response.json();
      //
      // 2. Hydrate Nested Field Paths directly into React Hook Form:
      //    If errData.fieldErrors is present, iterate over each [path, message]:
      //      for (const [fieldPath, message] of Object.entries(errData.fieldErrors)) {
      //        setError(fieldPath as any, {
      //          type: 'server',
      //          message,
      //        });
      //      }
      //    Notice how paths like "allocations.1.resourceId" or "shiftCode"
      //    automatically highlight the exact DOM input and preserve all user-typed data!
      //
      // 3. Handle 201 Success:
      //    Parse ShiftSubmitSuccessResponse and update setSubmitSuccess(resData).
      // ----------------------------------------------------------------------

      if (!response.ok) {
        const errorData: ServerValidationErrorResponse = await response.json();
        setSubmitError(errorData.message || 'Submission failed');

        if (errorData.fieldErrors) {
          for (const [fieldPath, message] of Object.entries(errorData.fieldErrors)) {
            setError(fieldPath as any, {
              type: 'server',
              message,
            });
          }
        }
        return false;
      }

      const successData: ShiftSubmitSuccessResponse = await response.json();
      setSubmitSuccess(successData);
      return true;
    } catch (err: any) {
      setSubmitError(err.message || 'Network error communicating with backend');
      return false;
    } finally {
      setIsSubmitting(false);
    }
  };

  return {
    isSubmitting,
    submitError,
    submitSuccess,
    submitWizard,
    resetSubmitState,
  };
}
