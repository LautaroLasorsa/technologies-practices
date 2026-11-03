import React from 'react';
import {
  useFieldArray,
  useWatch,
  type Control,
  type UseFormRegister,
  type FieldErrors,
} from 'react-hook-form';
import type { ShiftWizardInput } from '../../shared/schemas';

interface AllocationArrayFieldsProps {
  control: Control<ShiftWizardInput>;
  register: UseFormRegister<ShiftWizardInput>;
  errors: FieldErrors<ShiftWizardInput>;
}

/**
 * Isolated Component: Subscribes to allocation changes via useWatch.
 * Because useWatch is called HERE and NOT in the top-level form wizard,
 * updates only re-render this badge, maintaining O(1) input performance!
 */
export function AllocationMetricsBadge({ control }: { control: Control<ShiftWizardInput> }) {
  // useWatch isolates subscription updates to this specific component
  const allocations = useWatch({
    control,
    name: 'allocations',
    defaultValue: [],
  });

  const totalHours = allocations.reduce((sum, item) => sum + (Number(item?.allocatedHours) || 0), 0);
  const totalLoad = allocations.reduce((sum, item) => sum + (Number(item?.loadPercentage) || 0), 0);

  return (
    <div className="metrics-badge">
      <div className="metric-chip">
        <span className="metric-label">Allocated Resources:</span>
        <span className="metric-value">{allocations.length}</span>
      </div>
      <div className="metric-chip">
        <span className="metric-label">Total Hours:</span>
        <span className={`metric-value ${totalHours > 16 ? 'text-warning' : ''}`}>
          {totalHours.toFixed(1)} hrs
        </span>
      </div>
      <div className="metric-chip">
        <span className="metric-label">Aggregate Load:</span>
        <span className={`metric-value ${totalLoad > 200 ? 'text-danger' : ''}`}>
          {totalLoad}%
        </span>
      </div>
    </div>
  );
}

export function AllocationArrayFields({
  control,
  register,
  errors,
}: AllocationArrayFieldsProps) {
  // --------------------------------------------------------------------------
  // TODO(human) [Exercise 2]: Dynamic Field Array & Performance Isolation
  // --------------------------------------------------------------------------
  // Learning Objectives:
  // 1. Initialize `useFieldArray`:
  //    - Bind it to control and name: 'allocations'.
  //    - Extract `fields`, `append`, and `remove`.
  //
  // 2. Controlled Row Rendering via Unique Field IDs:
  //    - Map over `fields` and use `key={field.id}` (CRITICAL: never use index as key!).
  //    - Using field.id allows React Hook Form to correctly track DOM inputs when
  //      items are inserted, deleted, or reordered without resetting sibling values.
  //
  // 3. Register Nested Field Paths:
  //    - For each field row at `index`, register inputs to:
  //      `allocations.${index}.resourceId`
  //      `allocations.${index}.role`
  //      `allocations.${index}.allocatedHours` (with { valueAsNumber: true })
  //      `allocations.${index}.loadPercentage` (with { valueAsNumber: true })
  //
  // 4. Render Field-Specific Errors:
  //    - Check errors.allocations?.[index]?.resourceId?.message and highlight the field.
  //    - Note how Exercise 3's backend error hydration will directly light up these
  //      exact nested inputs via the identical path format!
  //
  // 5. Action Buttons:
  //    - Implement an "Add Resource" button invoking append({ ...defaults }).
  //    - Implement a "Remove" button per row invoking remove(index).
  // --------------------------------------------------------------------------

  const { fields, append, remove } = useFieldArray({
    control,
    name: 'allocations',
  });

  const handleAddDefault = () => {
    append({
      resourceId: `RES-ROBOT-${String(fields.length + 1).padStart(2, '0')}`,
      role: 'primary_operator',
      allocatedHours: 8,
      loadPercentage: 100,
    });
  };

  return (
    <div className="field-array-container">
      <div className="array-header">
        <div>
          <h3>Step 2: Dynamic Resource Allocations</h3>
          <p className="section-subtitle">
            Configure robotics, equipment, and technician rosters. Changes here use isolated
            subscriptions.
          </p>
        </div>
        <button
          type="button"
          onClick={handleAddDefault}
          className="btn btn-secondary btn-sm"
        >
          + Add Resource
        </button>
      </div>

      {/* Render the isolated subscription metrics badge */}
      <AllocationMetricsBadge control={control} />

      {/* Array-level error (e.g. "At least one resource must be allocated") */}
      {errors.allocations?.message && (
        <div className="alert alert-danger" role="alert">
          {errors.allocations.message}
        </div>
      )}

      {fields.length === 0 ? (
        <div className="empty-state">
          No resources allocated yet. Click <strong>+ Add Resource</strong> to add equipment.
        </div>
      ) : (
        <div className="allocation-list">
          {fields.map((field, index) => {
            const rowErrors = errors.allocations?.[index];

            return (
              <div key={field.id} className="allocation-card">
                <div className="card-topline">
                  <span className="row-index">Resource #{index + 1}</span>
                  <button
                    type="button"
                    onClick={() => remove(index)}
                    className="btn-icon-danger"
                    title="Remove resource"
                  >
                    &times;
                  </button>
                </div>

                <div className="grid grid-cols-4">
                  {/* Resource ID Input */}
                  <div className="form-group">
                    <label className="form-label">
                      Resource ID <span className="req">*</span>
                    </label>
                    <input
                      type="text"
                      className={`form-input ${rowErrors?.resourceId ? 'input-error' : ''}`}
                      placeholder="e.g. RES-ROBOT-01"
                      {...register(`allocations.${index}.resourceId` as const)}
                    />
                    {rowErrors?.resourceId && (
                      <span className="field-error-text">
                        {rowErrors.resourceId.message}
                      </span>
                    )}
                  </div>

                  {/* Role Select */}
                  <div className="form-group">
                    <label className="form-label">
                      Role <span className="req">*</span>
                    </label>
                    <select
                      className={`form-select ${rowErrors?.role ? 'input-error' : ''}`}
                      {...register(`allocations.${index}.role` as const)}
                    >
                      <option value="primary_operator">Primary Operator</option>
                      <option value="secondary_support">Secondary Support</option>
                      <option value="supervisor">Supervisor</option>
                    </select>
                    {rowErrors?.role && (
                      <span className="field-error-text">{rowErrors.role.message}</span>
                    )}
                  </div>

                  {/* Allocated Hours */}
                  <div className="form-group">
                    <label className="form-label">
                      Hours (0.5 - 12) <span className="req">*</span>
                    </label>
                    <input
                      type="number"
                      step="0.5"
                      className={`form-input ${rowErrors?.allocatedHours ? 'input-error' : ''}`}
                      {...register(`allocations.${index}.allocatedHours` as const, {
                        valueAsNumber: true,
                      })}
                    />
                    {rowErrors?.allocatedHours && (
                      <span className="field-error-text">
                        {rowErrors.allocatedHours.message}
                      </span>
                    )}
                  </div>

                  {/* Load Percentage */}
                  <div className="form-group">
                    <label className="form-label">
                      Load % (1 - 100) <span className="req">*</span>
                    </label>
                    <input
                      type="number"
                      step="5"
                      className={`form-input ${rowErrors?.loadPercentage ? 'input-error' : ''}`}
                      {...register(`allocations.${index}.loadPercentage` as const, {
                        valueAsNumber: true,
                      })}
                    />
                    {rowErrors?.loadPercentage && (
                      <span className="field-error-text">
                        {rowErrors.loadPercentage.message}
                      </span>
                    )}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
