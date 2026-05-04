#ifndef PLANET_CURVATURE_HLSL
#define PLANET_CURVATURE_HLSL

// Set every frame from OctreeGrid via Shader.SetGlobal* (shared by FarDistanceHeightmap + VoxelBlocks).
float _PlanetCurvatureRadius;
float4 _CurvatureRefWS;
// Altitude boost: strengthens sagitta when the viewer is high above nominal surface (Y orbit / flight).
float _CurvatureViewWorldY;
float _CurvatureSurfaceWorldY;
float _CurvatureAltitudeAmplify;
float _CurvatureAltitudeRampStart;
float _CurvatureAltitudeLogKnee;

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

// Full world-space sagitta with optional altitude-dependent multiplier (log so it keeps growing in orbit).
float PlanetCurvatureWorldDropY(float3 vtxWS)
{
    float baseDrop = SphericalCurvatureDropY(vtxWS.xz, _CurvatureRefWS.xz, _PlanetCurvatureRadius);
    if (_PlanetCurvatureRadius < 1.0)
        return 0.0;
    if (_CurvatureAltitudeAmplify <= 0.0)
        return baseDrop;

    float hEff = max(0.0, _CurvatureViewWorldY - _CurvatureSurfaceWorldY - _CurvatureAltitudeRampStart);
    if (hEff <= 0.0)
        return baseDrop;

    float u = log(1.0 + hEff / max(_CurvatureAltitudeLogKnee, 1.0));
    float boost = min(1.0 + _CurvatureAltitudeAmplify * u, 22.0);
    return baseDrop * boost;
}

#endif
