using System.Runtime.CompilerServices;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;

public static class MeshingUtility
{
    /// <summary>
    /// Writes mesh data into a Mesh.MeshDataArray slot.
    ///
    /// VERTEX STREAMS
    /// ──────────────
    /// Stream 0 — Position  (float3)
    /// Stream 1 — Normal    (float3)  flat per-face, written by NodeJob
    /// Stream 2 — TexCoord0 (float2)  xy = local UV (0–1), written by NodeJob
    /// Stream 3 — TexCoord1 (float2)  x  = blockId (float-cast byte),
    ///                                y  = texArray layer index for this face
    ///
    /// The VoxelBlocks shader reads TexCoord0 for UV and TexCoord1.x for the
    /// texture array layer to sample.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void ApplyMesh(
        ref Mesh.MeshDataArray meshDataArray,
        ref NativeArray<uint> indices,
        ref NativeArray<float3> vertices,
        ref NativeArray<float3> normals,
        ref NativeArray<float2> uvs,        // local face UVs  (TexCoord0)
        ref NativeArray<float2> blockData,  // x=blockId y=texLayer (TexCoord1)
        ref Bounds bounds,
        IndexFormat indexFormat = IndexFormat.UInt32)
    {
        Mesh.MeshData meshData = meshDataArray[0];

        var vertexAttributes = new NativeArray<VertexAttributeDescriptor>(
            4, Allocator.Temp, NativeArrayOptions.UninitializedMemory);

        vertexAttributes[0] = new VertexAttributeDescriptor(
            VertexAttribute.Position, dimension: 3, stream: 0);
        vertexAttributes[1] = new VertexAttributeDescriptor(
            VertexAttribute.Normal, dimension: 3, stream: 1);
        vertexAttributes[2] = new VertexAttributeDescriptor(
            VertexAttribute.TexCoord0, dimension: 2, stream: 2);  // UV
        vertexAttributes[3] = new VertexAttributeDescriptor(
            VertexAttribute.TexCoord1, dimension: 2, stream: 3);  // blockId + texLayer

        meshData.SetVertexBufferParams(vertices.Length, vertexAttributes);

        meshData.GetVertexData<float3>(0).CopyFrom(vertices);
        meshData.GetVertexData<float3>(1).CopyFrom(normals);
        meshData.GetVertexData<float2>(2).CopyFrom(uvs);
        meshData.GetVertexData<float2>(3).CopyFrom(blockData);

        meshData.SetIndexBufferParams(indices.Length, indexFormat);
        meshData.GetIndexData<uint>().CopyFrom(indices);

        meshData.subMeshCount = 1;
        meshData.SetSubMesh(0, new SubMeshDescriptor(0, indices.Length)
        {
            bounds = bounds,
            vertexCount = vertices.Length,
        }, MeshUpdateFlags.DontRecalculateBounds);
    }
}