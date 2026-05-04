Shader "Voxel Void/FarDistanceHeightmap"
{
    Properties
    {
        _HeightTex ("Height Texture", 2D) = "gray" {}
        _FarColor ("Far Color", Color) = (0.40, 0.42, 0.45, 1)
        _HorizonColor ("Horizon Fog Color", Color) = (0.55, 0.58, 0.62, 1)
        _SurfaceBaseHeight ("Surface Base Height", Float) = 0
        _SurfaceAmplitude ("Surface Amplitude", Float) = 300
        _NearFadeStart ("Near Fade Start", Float) = 1800
        _NearFadeEnd ("Near Fade End", Float) = 3600
        _FarFadeStart ("Far Fade Start", Float) = 20000
        _FarFadeEnd ("Far Fade End", Float) = 28000
        _PlayerHoleInner ("Player 3D Hole Inner", Float) = 400
        _PlayerHoleOuter ("Player 3D Hole Outer", Float) = 3600
        _UndergroundFadeMode ("Underground Fade Mode (0=off,1=below bedrock Y)", Float) = 0
        _WorldBedrockY ("World Bedrock Y (min subsurface)", Float) = -500
        _UndergroundFadeStart ("Underground Fade Start (m below bedrock when mode=1)", Float) = 32
        _UndergroundFadeEnd ("Underground Fade End (m below bedrock when mode=1)", Float) = 400
        _FogHorizonMix ("Fog Horizon Mix", Range(0,1)) = 0.65
        _UnderlayYOffset ("Underlay Y Offset (optional)", Float) = 0
        // Far depth: reversed-Z uses z=w*epsilon; classic uses z=w*(1-epsilon). Tune on material / OctreeGrid.
        _BackgroundClipEpsilon ("Background clip epsilon", Range(0.000001, 0.001)) = 0.00001
    }

    SubShader
    {
        Tags
        {
            "RenderType" = "Opaque"
            "RenderPipeline" = "UniversalPipeline"
            "Queue" = "Geometry-50"
        }

        Pass
        {
            Name "ForwardUnlit"
            Tags { "LightMode" = "UniversalForward" }

            Cull Back
            ZWrite Off
            ZTest LEqual

            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma multi_compile_fog

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/UnityInput.hlsl"
            #include "Include/PlanetCurvature.hlsl"

            TEXTURE2D(_HeightTex);
            SAMPLER(sampler_HeightTex);

            CBUFFER_START(UnityPerMaterial)
                float4 _FarColor;
                float4 _HorizonColor;
                float4 _WorldSizeXZ;
                float4 _PlayerWorldPos;
                float4 _FarCenterWS;
                float _SurfaceBaseHeight;
                float _SurfaceAmplitude;
                float _UndergroundFadeMode;
                float _WorldBedrockY;
                float _NearFadeStart;
                float _NearFadeEnd;
                float _FarFadeStart;
                float _FarFadeEnd;
                float _PlayerHoleInner;
                float _PlayerHoleOuter;
                float _UndergroundFadeStart;
                float _UndergroundFadeEnd;
                float _FogHorizonMix;
                float _UnderlayYOffset;
                float _BackgroundClipEpsilon;
            CBUFFER_END

            struct Attributes
            {
                float3 positionOS : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct Varyings
            {
                float4 positionCS : SV_POSITION;
                float3 positionWS : TEXCOORD0;
                float fogFactor : TEXCOORD1;
                float visibility : TEXCOORD2;
            };

            static float PosMod(float x, float m)
            {
                float r = x % m;
                return r < 0.0 ? r + m : r;
            }

            Varyings vert(Attributes IN)
            {
                Varyings OUT;

                float3 ws = float3(
                    _FarCenterWS.x + IN.positionOS.x,
                    0.0,
                    _FarCenterWS.z + IN.positionOS.z);
                float wx = ws.x;
                float wz = ws.z;
                float2 worldSize = max(_WorldSizeXZ.xy, float2(1.0, 1.0));
                float2 uvWrap = float2(
                    PosMod(wx, worldSize.x) / worldSize.x,
                    PosMod(wz, worldSize.y) / worldSize.y);

                float h01 = SAMPLE_TEXTURE2D_LOD(_HeightTex, sampler_HeightTex, uvWrap, 0).r;
                float terrainY = _SurfaceBaseHeight - _SurfaceAmplitude + h01 * (2.0 * _SurfaceAmplitude);
                ws.y = terrainY - max(0.0, _UnderlayYOffset);
                ws.y -= PlanetCurvatureWorldDropY(ws);

                float4 clipPos = TransformWorldToHClip(ws);

                // Skybox-style depth: push to far plane in NDC without ZWrite (octrees draw later on top).
                float e = max(_BackgroundClipEpsilon, 1e-7);
#if UNITY_REVERSED_Z
                clipPos.z = clipPos.w * e;
#else
                clipPos.z = clipPos.w * (1.0 - e);
#endif

                OUT.positionCS = clipPos;
                OUT.positionWS = ws;
                OUT.fogFactor = ComputeFogFactor(clipPos.z);

                float3 player = _PlayerWorldPos.xyz;
                float d = distance(ws.xz, player.xz);
                float nearVis = smoothstep(_NearFadeStart, max(_NearFadeStart + 1e-3, _NearFadeEnd), d);
                float farVis = 1.0 - smoothstep(_FarFadeStart, max(_FarFadeStart + 1e-3, _FarFadeEnd), d);

                float d3 = distance(ws, player);
                float innerH = max(0.0, _PlayerHoleInner);
                float outerH = max(innerH + 1e-3, _PlayerHoleOuter);
                float holeVis = smoothstep(innerH, outerH, d3);

                float underVis = 1.0;
                if (_UndergroundFadeMode > 0.5)
                {
                    float belowBedrock = max(0.0, _WorldBedrockY - player.y);
                    float u0 = max(0.0, _UndergroundFadeStart);
                    float u1 = max(u0 + 1e-3, _UndergroundFadeEnd);
                    underVis = 1.0 - smoothstep(u0, u1, belowBedrock);
                }

                OUT.visibility = saturate(nearVis * farVis * holeVis * underVis);
                return OUT;
            }

            half4 frag(Varyings IN) : SV_Target
            {
                float v = saturate(IN.visibility);
                half3 baseCol = _FarColor.rgb;
                float horizonT = (1.0 - v) * saturate(_FogHorizonMix);
                half3 col = lerp(baseCol, _HorizonColor.rgb, horizonT);
                col = MixFog(col, IN.fogFactor);
                return half4(col, 1.0);
            }
            ENDHLSL
        }
    }
}
