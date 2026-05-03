#ifndef PLANET_CURVATURE_HLSL
#define PLANET_CURVATURE_HLSL

// Set every frame from OctreeGrid via Shader.SetGlobal* (shared by FarDistanceHeightmap + VoxelBlocks).
float _PlanetCurvatureRadius;
float4 _CurvatureRefWS;

float SphericalCurvatureDropY(float2 vtxXZ, float2 refXZ, float R)
{
    if (R < 1.0)
        return 0.0;
    float dst = distance(vtxXZ, refXZ);
    float d = min(dst, R * (1.0 - 1e-5));
    float inner = R * R - d * d;
    inner = max(inner, 0.0);
    float s = sqrt(inner);
    return (d * d) / max(R + s, 1e-6);
}

#endif
