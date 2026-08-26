#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

extern "C" cudaError_t jkf_initialize_custom_fft_twiddles();

extern "C" void jkf_launch_custom_fft_f30_forward(
    const float2* image, const float* scaling, float2* transposed_workspace,
    half* n16_output, cudaStream_t stream);

extern "C" void jkf_launch_custom_fft_f30_forward_split2(
    const float2* image, const float* scaling, float2* transposed_workspace,
    half* n16_high_output, half* n16_residual_output, cudaStream_t stream);

extern "C" void jkf_launch_custom_fft_f30_forward_split2_debug(
    const float2* image, const float* scaling, float2* transposed_workspace,
    half* n16_high_output, half* n16_residual_output,
    float2* fp32_endpoint_output, cudaStream_t stream);

extern "C" void jkf_launch_custom_fft_f30_forward_sense(
    const float2* image, const float2* sensitivity_maps,
    const float* scaling, float2* transposed_workspace,
    half* n16_output, cudaStream_t stream);

extern "C" void jkf_launch_custom_fft_f30_forward_sense_split2(
    const float2* image, const float2* sensitivity_maps,
    const float* scaling, float2* transposed_workspace,
    half* n16_high_output, half* n16_residual_output, cudaStream_t stream);

extern "C" void jkf_launch_custom_ifft_f31_adjoint(
    const float2* grid_input, float2* transposed_workspace,
    const float* scaling, const float* scaling_tiled, float2* image_output,
    cudaStream_t stream);
