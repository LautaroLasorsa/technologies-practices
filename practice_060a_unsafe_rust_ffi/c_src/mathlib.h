/**
 * mathlib.h — A small C math/utility library for FFI exercises.
 *
 * This header declares functions that Rust will call via FFI.
 * bindgen will also parse this file to auto-generate Rust bindings.
 *
 * All functions use standard C types and calling conventions.
 * Structs use explicit field ordering (Rust must use #[repr(C)] to match).
 */

#ifndef MATHLIB_H
#define MATHLIB_H

#include <stddef.h>  /* size_t */
#include <stdint.h>  /* int32_t, uint32_t */

/* ── Constants ────────────────────────────────────────────────────── */

/** Maximum vector dimension supported by this library. */
#define MATHLIB_MAX_DIM 1024

/** Version of the mathlib API. */
#define MATHLIB_VERSION 1

/* ── Structs ──────────────────────────────────────────────────────── */

/**
 * A 3D point in Cartesian coordinates.
 * Layout: x at offset 0, y at offset 8, z at offset 16. Total: 24 bytes.
 * Rust must use #[repr(C)] for a matching struct.
 */
typedef struct {
    double x;
    double y;
    double z;
} Point3D;

/**
 * Result of a statistical computation.
 * Contains both the computed value and an error code.
 */
typedef struct {
    double value;
    int32_t error_code;  /* 0 = success, non-zero = error */
    /* 4 bytes padding here on most platforms (alignment of double = 8) */
} StatResult;

/* ── Basic Arithmetic ─────────────────────────────────────────────── */

/**
 * Add two 32-bit integers.
 * This is the simplest possible FFI function: no pointers, no allocation.
 */
int32_t mathlib_add(int32_t a, int32_t b);

/**
 * Multiply two 32-bit integers.
 */
int32_t mathlib_multiply(int32_t a, int32_t b);

/* ── Array Operations ─────────────────────────────────────────────── */

/**
 * Compute the dot product of two arrays of doubles.
 *
 * @param a     Pointer to first array (must not be NULL)
 * @param b     Pointer to second array (must not be NULL)
 * @param len   Number of elements in each array
 * @return      Sum of a[i] * b[i] for i in [0, len)
 *
 * Caller responsibility: both arrays must have at least `len` elements.
 * Passing a shorter array is undefined behavior (buffer over-read).
 */
double mathlib_dot_product(const double *a, const double *b, size_t len);

/**
 * Sum all elements in an int32_t array.
 *
 * @param arr   Pointer to array (must not be NULL)
 * @param len   Number of elements
 * @return      Sum of all elements
 */
int64_t mathlib_sum_array(const int32_t *arr, size_t len);

/* ── String Operations ────────────────────────────────────────────── */

/**
 * Format a vector of doubles as a string: "[1.00, 2.00, 3.00]".
 *
 * @param arr   Pointer to array of doubles
 * @param len   Number of elements
 * @return      Heap-allocated string (caller must free with free())
 *
 * OWNERSHIP: The returned char* is malloc'd. The CALLER is responsible
 * for calling free() on it. Failing to do so leaks memory.
 * This is a common C API pattern that Rust must handle carefully.
 */
char *mathlib_format_vector(const double *arr, size_t len);

/* ── Struct Operations ────────────────────────────────────────────── */

/**
 * Compute the Euclidean magnitude (length) of a 3D point/vector.
 *
 * @param p   Pointer to a Point3D struct
 * @return    sqrt(x*x + y*y + z*z)
 */
double mathlib_point_magnitude(const Point3D *p);

/**
 * Add two 3D points component-wise and return the result by value.
 *
 * Returning a struct by value from C is well-defined and works across
 * FFI — the compiler handles the ABI details (often via hidden pointer).
 */
Point3D mathlib_point_add(Point3D a, Point3D b);

/**
 * Compute basic statistics (mean) of an array, returning a StatResult.
 *
 * @param arr   Pointer to array of doubles (NULL returns error)
 * @param len   Number of elements (0 returns error)
 * @return      StatResult with mean value and error code
 */
StatResult mathlib_mean(const double *arr, size_t len);

/* ── Callback Operations ──────────────────────────────────────────── */

/**
 * Apply a transformation function to each element of an array.
 *
 * @param arr       Array of doubles (modified in-place)
 * @param len       Number of elements
 * @param transform Function pointer: takes a double, returns a double
 *
 * This demonstrates C calling a Rust function pointer (callback).
 * The callback must use the C calling convention.
 */
void mathlib_transform_array(double *arr, size_t len,
                             double (*transform)(double));

#endif /* MATHLIB_H */
