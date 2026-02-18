/**
 * mathlib.c — Implementation of the math/utility library for FFI exercises.
 *
 * This file is compiled by the `cc` crate in build.rs and linked into
 * the Rust binary as a static library.
 *
 * All functions follow C calling conventions and use standard C types.
 * Memory allocation uses malloc/free (NOT Rust's allocator).
 */

#include "mathlib.h"
#include <math.h>    /* sqrt */
#include <stdio.h>   /* snprintf */
#include <stdlib.h>  /* malloc, free */
#include <string.h>  /* memset */

/* ── Basic Arithmetic ─────────────────────────────────────────────── */

int32_t mathlib_add(int32_t a, int32_t b) {
    return a + b;
}

int32_t mathlib_multiply(int32_t a, int32_t b) {
    return a * b;
}

/* ── Array Operations ─────────────────────────────────────────────── */

double mathlib_dot_product(const double *a, const double *b, size_t len) {
    double sum = 0.0;
    for (size_t i = 0; i < len; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

int64_t mathlib_sum_array(const int32_t *arr, size_t len) {
    int64_t sum = 0;
    for (size_t i = 0; i < len; i++) {
        sum += arr[i];
    }
    return sum;
}

/* ── String Operations ────────────────────────────────────────────── */

char *mathlib_format_vector(const double *arr, size_t len) {
    /*
     * Allocate a buffer large enough for the formatted output.
     * Each double can be at most ~24 chars (with formatting), plus
     * separators ", " (2 chars each), brackets "[]", and null terminator.
     */
    size_t buf_size = len * 26 + 3;  /* generous estimate */
    char *buf = (char *)malloc(buf_size);
    if (!buf) return NULL;

    size_t pos = 0;
    buf[pos++] = '[';

    for (size_t i = 0; i < len; i++) {
        if (i > 0) {
            buf[pos++] = ',';
            buf[pos++] = ' ';
        }
        int written = snprintf(buf + pos, buf_size - pos, "%.2f", arr[i]);
        if (written > 0) {
            pos += (size_t)written;
        }
    }

    buf[pos++] = ']';
    buf[pos] = '\0';

    return buf;  /* CALLER must free() this! */
}

/* ── Struct Operations ────────────────────────────────────────────── */

double mathlib_point_magnitude(const Point3D *p) {
    return sqrt(p->x * p->x + p->y * p->y + p->z * p->z);
}

Point3D mathlib_point_add(Point3D a, Point3D b) {
    Point3D result;
    result.x = a.x + b.x;
    result.y = a.y + b.y;
    result.z = a.z + b.z;
    return result;
}

StatResult mathlib_mean(const double *arr, size_t len) {
    StatResult result;
    memset(&result, 0, sizeof(result));

    if (arr == NULL || len == 0) {
        result.value = 0.0;
        result.error_code = 1;  /* error: invalid input */
        return result;
    }

    double sum = 0.0;
    for (size_t i = 0; i < len; i++) {
        sum += arr[i];
    }

    result.value = sum / (double)len;
    result.error_code = 0;  /* success */
    return result;
}

/* ── Callback Operations ──────────────────────────────────────────── */

void mathlib_transform_array(double *arr, size_t len,
                             double (*transform)(double)) {
    for (size_t i = 0; i < len; i++) {
        arr[i] = transform(arr[i]);
    }
}
