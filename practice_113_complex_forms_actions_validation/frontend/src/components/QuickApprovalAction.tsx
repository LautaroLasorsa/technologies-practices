import React, { useActionState } from 'react';
import { useFormStatus } from 'react-dom';
import type { QuickApprovalResponse } from '../../shared/schemas';

interface QuickApprovalActionProps {
  shiftId: string;
  onApproved?: (response: QuickApprovalResponse) => void;
}

interface ApprovalActionState {
  status: 'idle' | 'success' | 'error';
  message: string | null;
  data?: QuickApprovalResponse | null;
}

const initialState: ApprovalActionState = {
  status: 'idle',
  message: null,
  data: null,
};

/**
 * Child component that consumes React 19's useFormStatus().
 * Reads the pending status directly from the parent <form> context without prop drilling.
 */
function SubmitButton() {
  const { pending } = useFormStatus();

  return (
    <button
      type="submit"
      disabled={pending}
      className={`btn btn-primary ${pending ? 'btn-loading' : ''}`}
    >
      {pending ? (
        <>
          <span className="spinner" /> Approving via Action...
        </>
      ) : (
        'Sign & Approve Shift'
      )}
    </button>
  );
}

export function QuickApprovalAction({ shiftId, onApproved }: QuickApprovalActionProps) {
  // --------------------------------------------------------------------------
  // TODO(human) [Exercise 4: React 19 Actions (useActionState + useFormStatus)]
  // --------------------------------------------------------------------------
  // Learning Objectives:
  // 1. Understand React 19 Action Transition Semantics:
  //    React 19 Actions manage async server mutations automatically inside a
  //    React Transition. Notice how useActionState gives us:
  //    [state, formAction, isPending]
  //
  // 2. Author the Action Function:
  //    The action receives `(prevState, formData: FormData) => Promise<NewState>`.
  //    Extract form inputs via formData.get('supervisorNotes') and formData.get('overrideCode').
  //    Post to `/api/shifts/${shiftId}/quick-approval`.
  //
  // 3. Compare Architecture:
  //    Notice how concise this is for atomic RPC operations compared to setting up
  //    controlled state or complex RHF instances. RHF excels at large multi-field
  //    dynamic arrays; React 19 Actions excel at atomic transactional mutations.
  // --------------------------------------------------------------------------

  async function approvalAction(
    _prevState: ApprovalActionState,
    formData: FormData
  ): Promise<ApprovalActionState> {
    const supervisorNotes = formData.get('supervisorNotes') as string;
    const overrideCode = (formData.get('overrideCode') as string) || undefined;

    try {
      const res = await fetch(`/api/shifts/${encodeURIComponent(shiftId)}/quick-approval`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ supervisorNotes, overrideCode }),
      });

      const json = await res.json();
      if (!res.ok) {
        return {
          status: 'error',
          message: json.message || 'Approval rejected by server.',
        };
      }

      const responseData = json as QuickApprovalResponse;
      if (onApproved) {
        onApproved(responseData);
      }

      return {
        status: 'success',
        message: responseData.message,
        data: responseData,
      };
    } catch (err: any) {
      return {
        status: 'error',
        message: err.message || 'Network failure during action execution.',
      };
    }
  }

  const [state, formAction, isPending] = useActionState(approvalAction, initialState);

  return (
    <div className="quick-approval-card">
      <div className="approval-header">
        <h4>Exercise 4: Supervisor Quick-Signoff Action</h4>
        <span className="badge-react19">React 19 Native Action</span>
      </div>
      <p className="approval-desc">
        Target Shift: <code>{shiftId}</code>. Demonstrates transition-driven execution with{' '}
        <code>useActionState</code> and <code>useFormStatus</code>.
      </p>

      <form action={formAction} className="approval-form">
        <div className="form-group">
          <label className="form-label">
            Supervisor Sign-off Notes <span className="req">*</span>
          </label>
          <input
            type="text"
            name="supervisorNotes"
            required
            defaultValue="Line clear and safety interlocks verified."
            placeholder="Minimum 5 characters notes..."
            className="form-input"
            disabled={isPending}
          />
        </div>

        <div className="form-group">
          <label className="form-label">
            Emergency Manager Override Code (Optional)
          </label>
          <input
            type="text"
            name="overrideCode"
            placeholder="e.g. OVERRIDE-2026"
            className="form-input"
            disabled={isPending}
          />
        </div>

        <div className="approval-actions">
          <SubmitButton />
          {isPending && <span className="text-muted">Transition in progress...</span>}
        </div>
      </form>

      {state.status === 'error' && (
        <div className="alert alert-danger mt-3">
          <strong>Action Failed:</strong> {state.message}
        </div>
      )}

      {state.status === 'success' && (
        <div className="alert alert-success mt-3">
          <strong>Success:</strong> {state.message}
          <div className="small text-muted mt-1">
            Approved by: {state.data?.approvedBy} at {state.data?.timestamp}
          </div>
        </div>
      )}
    </div>
  );
}
