import React, { useState, useEffect } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import {
  shiftWizardSchema,
  type ShiftWizardInput,
} from '../../shared/schemas';
import { AllocationArrayFields } from './components/AllocationArrayFields';
import { QuickApprovalAction } from './components/QuickApprovalAction';
import { useWizardSubmit } from './hooks/useWizardSubmit';

export default function App() {
  const [currentStep, setCurrentStep] = useState<1 | 2>(1);
  const [bookedState, setBookedState] = useState<{
    activeShiftCodes: string[];
    bookedResources: Array<{
      resourceId: string;
      shiftCode: string;
      startTime: string;
      endTime: string;
    }>;
  } | null>(null);

  // Initialize React Hook Form with the shared Zod schema
  const {
    register,
    control,
    handleSubmit,
    watch,
    setError,
    trigger,
    formState: { errors, isValid, isDirty, isSubmitting: formIsSubmitting },
    reset,
  } = useForm<ShiftWizardInput>({
    resolver: zodResolver(shiftWizardSchema),
    mode: 'onBlur',
    defaultValues: {
      shiftCode: 'SHIFT-2026-003',
      department: 'assembly',
      shiftType: 'standard',
      startTime: '08:00',
      endTime: '16:30',
      handoverSupervisor: '',
      notes: 'Initial production line setup.',
      allocations: [
        {
          resourceId: 'RES-ROBOT-01', // Will trigger Exercise 3 collision if standard time overlaps!
          role: 'primary_operator',
          allocatedHours: 8,
          loadPercentage: 100,
        },
      ],
    },
  });

  // Watch shiftType to toggle conditional fields (overnight supervisor)
  const shiftType = watch('shiftType');

  // Exercise 3 submission hook
  const { isSubmitting, submitError, submitSuccess, submitWizard, resetSubmitState } =
    useWizardSubmit(setError);

  // Load backend booked resources on mount to visualize collision targets
  const fetchBookedWindows = async () => {
    try {
      const res = await fetch('/api/shifts/booked-windows');
      if (res.ok) {
        const data = await res.json();
        setBookedState(data);
      }
    } catch {
      // Backend not running yet or starting up
    }
  };

  useEffect(() => {
    fetchBookedWindows();
  }, [submitSuccess]);

  const handleNextStep = async () => {
    // Validate Step 1 fields before advancing
    const step1Valid = await trigger([
      'shiftCode',
      'department',
      'shiftType',
      'startTime',
      'endTime',
      'handoverSupervisor',
    ]);
    if (step1Valid) {
      setCurrentStep(2);
    }
  };

  const onValidSubmit = async (data: ShiftWizardInput) => {
    const success = await submitWizard(data);
    if (success) {
      fetchBookedWindows();
    }
  };

  return (
    <div className="app-container">
      {/* Top Header */}
      <header className="app-header">
        <div className="header-content">
          <div className="badge-tech">Practice 113</div>
          <h1>Production Shift & Resource Allocation Wizard</h1>
          <p className="header-subtitle">
            Complex Forms, Action Workflows & Multi-Layer Validation (React 19 + RHF + Zod + Fastify)
          </p>
        </div>
      </header>

      <div className="app-main-grid">
        {/* Main Wizard Form Column */}
        <main className="form-column">
          {/* Stepper Navigation */}
          <div className="stepper-bar">
            <button
              type="button"
              className={`step-indicator ${currentStep === 1 ? 'step-active' : ''}`}
              onClick={() => setCurrentStep(1)}
            >
              <span className="step-num">1</span>
              <span className="step-text">Shift Header</span>
            </button>
            <div className="step-divider" />
            <button
              type="button"
              className={`step-indicator ${currentStep === 2 ? 'step-active' : ''}`}
              onClick={() => setCurrentStep(2)}
            >
              <span className="step-num">2</span>
              <span className="step-text">Resource Allocations</span>
            </button>
          </div>

          {/* Submission Alerts */}
          {submitError && (
            <div className="alert alert-danger" role="alert">
              <strong>Submission Error:</strong> {submitError}
            </div>
          )}

          {submitSuccess && (
            <div className="alert alert-success" role="alert">
              <strong>Shift Created Successfully!</strong>
              <div>Shift ID: <code>{submitSuccess.shiftId}</code></div>
              <p className="mt-1">{submitSuccess.message}</p>
            </div>
          )}

          <form onSubmit={handleSubmit(onValidSubmit)} noValidate>
            {/* Step 1: Shift Header */}
            {currentStep === 1 && (
              <section className="form-step-card">
                <h3>Step 1: Shift Header Information</h3>
                <p className="section-subtitle">
                  Configure scheduling window, department, and shift invariants.
                </p>

                <div className="grid grid-cols-2 mt-4">
                  {/* Shift Code */}
                  <div className="form-group">
                    <label className="form-label">
                      Shift Code <span className="req">*</span>
                    </label>
                    <input
                      type="text"
                      className={`form-input ${errors.shiftCode ? 'input-error' : ''}`}
                      placeholder="SHIFT-2026-001"
                      {...register('shiftCode')}
                    />
                    {errors.shiftCode && (
                      <span className="field-error-text">{errors.shiftCode.message}</span>
                    )}
                    <span className="field-help">Format: SHIFT-YYYY-NNN</span>
                  </div>

                  {/* Department */}
                  <div className="form-group">
                    <label className="form-label">
                      Department <span className="req">*</span>
                    </label>
                    <select
                      className={`form-select ${errors.department ? 'input-error' : ''}`}
                      {...register('department')}
                    >
                      <option value="assembly">Assembly</option>
                      <option value="machining">Machining</option>
                      <option value="packaging">Packaging</option>
                      <option value="quality_control">Quality Control</option>
                    </select>
                    {errors.department && (
                      <span className="field-error-text">{errors.department.message}</span>
                    )}
                  </div>
                </div>

                <div className="grid grid-cols-3 mt-3">
                  {/* Shift Type (Discriminator) */}
                  <div className="form-group">
                    <label className="form-label">
                      Shift Type <span className="req">*</span>
                    </label>
                    <select
                      className={`form-select ${errors.shiftType ? 'input-error' : ''}`}
                      {...register('shiftType')}
                    >
                      <option value="standard">Standard (Same-Day)</option>
                      <option value="overnight">Overnight (Crosses Midnight)</option>
                    </select>
                    {errors.shiftType && (
                      <span className="field-error-text">{errors.shiftType.message}</span>
                    )}
                  </div>

                  {/* Start Time */}
                  <div className="form-group">
                    <label className="form-label">
                      Start Time <span className="req">*</span>
                    </label>
                    <input
                      type="time"
                      className={`form-input ${errors.startTime ? 'input-error' : ''}`}
                      {...register('startTime')}
                    />
                    {errors.startTime && (
                      <span className="field-error-text">{errors.startTime.message}</span>
                    )}
                  </div>

                  {/* End Time */}
                  <div className="form-group">
                    <label className="form-label">
                      End Time <span className="req">*</span>
                    </label>
                    <input
                      type="time"
                      className={`form-input ${errors.endTime ? 'input-error' : ''}`}
                      {...register('endTime')}
                    />
                    {errors.endTime && (
                      <span className="field-error-text">{errors.endTime.message}</span>
                    )}
                  </div>
                </div>

                {/* Conditional Field: Overnight Handover Supervisor */}
                {shiftType === 'overnight' && (
                  <div className="form-group mt-3 highlight-box">
                    <label className="form-label">
                      Handover Supervisor (Overnight Requirement) <span className="req">*</span>
                    </label>
                    <input
                      type="text"
                      className={`form-input ${errors.handoverSupervisor ? 'input-error' : ''}`}
                      placeholder="e.g. Elena Rostova"
                      {...register('handoverSupervisor')}
                    />
                    {errors.handoverSupervisor && (
                      <span className="field-error-text">
                        {errors.handoverSupervisor.message}
                      </span>
                    )}
                  </div>
                )}

                <div className="form-group mt-3">
                  <label className="form-label">Operational Notes</label>
                  <textarea
                    rows={2}
                    className="form-textarea"
                    placeholder="Safety alerts, maintenance directives, or shift notes..."
                    {...register('notes')}
                  />
                  {errors.notes && (
                    <span className="field-error-text">{errors.notes.message}</span>
                  )}
                </div>

                <div className="wizard-footer mt-4">
                  <button
                    type="button"
                    onClick={handleNextStep}
                    className="btn btn-primary"
                  >
                    Next: Resource Allocations &rarr;
                  </button>
                </div>
              </section>
            )}

            {/* Step 2: Dynamic Resource Allocations */}
            {currentStep === 2 && (
              <section className="form-step-card">
                <AllocationArrayFields
                  control={control}
                  register={register}
                  errors={errors}
                />

                <div className="wizard-footer mt-4">
                  <button
                    type="button"
                    onClick={() => setCurrentStep(1)}
                    className="btn btn-outline"
                  >
                    &larr; Back to Step 1
                  </button>
                  <button
                    type="submit"
                    disabled={isSubmitting || formIsSubmitting}
                    className="btn btn-success"
                  >
                    {isSubmitting ? 'Validating & Submitting...' : 'Submit Shift Configuration'}
                  </button>
                </div>
              </section>
            )}
          </form>

          {/* Exercise 4: Atomic Quick Approval Action */}
          <section className="mt-4">
            <QuickApprovalAction
              shiftId={submitSuccess?.shiftId || 'SHIFT-2026-001'}
              onApproved={(res) => {
                fetchBookedWindows();
              }}
            />
          </section>
        </main>

        {/* Sidebar Diagnostics Column */}
        <aside className="sidebar-column">
          {/* Active Database State (Layer 2 Collisions Target) */}
          <div className="sidebar-card">
            <h4>Database State (Layer 2 Collisions)</h4>
            <p className="card-desc">
              Values registered in backend database. Submitting these triggers HTTP 422
              field-error hydration.
            </p>

            <div className="mock-data-section">
              <span className="sub-title">Active Shift Codes:</span>
              <ul className="pill-list">
                {bookedState?.activeShiftCodes.map((code) => (
                  <li key={code} className="pill-code">
                    {code}
                  </li>
                )) || <li className="text-muted">Loading or backend offline...</li>}
              </ul>
            </div>

            <div className="mock-data-section mt-3">
              <span className="sub-title">Booked Resource Windows:</span>
              <div className="booked-items-list">
                {bookedState?.bookedResources.map((item, idx) => (
                  <div key={idx} className="booked-item">
                    <strong>{item.resourceId}</strong> ({item.shiftCode})
                    <div className="small text-muted">{item.startTime} - {item.endTime}</div>
                  </div>
                )) || <div className="text-muted">Loading...</div>}
              </div>
            </div>
          </div>

          {/* Form State Live Telemetry */}
          <div className="sidebar-card mt-3">
            <h4>RHF State Telemetry</h4>
            <div className="telemetry-grid">
              <div className="telemetry-row">
                <span>Form isDirty:</span>
                <span className={isDirty ? 'text-success' : 'text-muted'}>
                  {isDirty ? 'Yes' : 'No'}
                </span>
              </div>
              <div className="telemetry-row">
                <span>Form isValid:</span>
                <span className={isValid ? 'text-success' : 'text-danger'}>
                  {isValid ? 'Yes' : 'No'}
                </span>
              </div>
              <div className="telemetry-row">
                <span>Active Error Count:</span>
                <span className={Object.keys(errors).length > 0 ? 'text-danger' : 'text-success'}>
                  {Object.keys(errors).length}
                </span>
              </div>
            </div>

            {Object.keys(errors).length > 0 && (
              <div className="error-telemetry-list mt-2">
                <span className="sub-title">Current Errors:</span>
                <pre className="json-pre">
                  {JSON.stringify(errors, (key, value) => {
                    if (key === 'ref') return undefined;
                    return value;
                  }, 2)}
                </pre>
              </div>
            )}
          </div>
        </aside>
      </div>
    </div>
  );
}
