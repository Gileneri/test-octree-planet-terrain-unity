Shader "Voxel Void/VoxelBlocks"
{
    Properties
    {
        _TextureArray ("Block Texture Array", 2DArray) = "" {}
        [HideInInspector] _TextureArray_ST ("", Vector) = (1,1,0,0)
        _Smoothness   ("Smoothness",  Range(0,1)) = 0.0
        _Metallic     ("Metallic",    Range(0,1)) = 0.0
    }

    SubShader
    {
        Tags
        {
            "RenderType"      = "Opaque"
            "RenderPipeline"  = "UniversalPipeline"
            "Queue"           = "Geometry"
        }

        // ── Forward Lit pass ─────────────────────────────────────────────
        Pass
        {
            Name "ForwardLit"
            Tags { "LightMode" = "UniversalForward" }

            Cull Back
            ZWrite On

            HLSLPROGRAM
            #pragma vertex   vert
            #pragma fragment frag

            // URP lighting keywords
            #pragma multi_compile _ _MAIN_LIGHT_SHADOWS _MAIN_LIGHT_SHADOWS_CASCADE
            #pragma multi_compile _ _ADDITIONAL_LIGHTS_VERTEX _ADDITIONAL_LIGHTS
            #pragma multi_compile _ _SHADOWS_SOFT
            #pragma multi_compile_fog

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Lighting.hlsl"
            #include "Include/PlanetCurvature.hlsl"

            // ── Texture array ────────────────────────────────────────────
            TEXTURE2D_ARRAY(_TextureArray);
            // Declaring as point sampler keeps pixel art textures crisp.
            // Unity matches sampler name to texture name automatically.
            SamplerState sampler_point_repeat_TextureArray;

            // ── Material properties ──────────────────────────────────────
            CBUFFER_START(UnityPerMaterial)
                float _Smoothness;
                float _Metallic;
            CBUFFER_END

            // ── Vertex input ─────────────────────────────────────────────
            struct Attributes
            {
                float3 positionOS : POSITION;    // stream 0
                float3 normalOS   : NORMAL;      // stream 1
                float2 uv         : TEXCOORD0;   // stream 2 — face UV 0-1
                float2 blockData  : TEXCOORD1;   // stream 3 — x=blockId y=texLayer
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            // ── Vertex to fragment ───────────────────────────────────────
            struct Varyings
            {
                float4 positionCS : SV_POSITION;
                float3 positionWS : TEXCOORD0;
                float3 normalWS   : TEXCOORD1;
                float2 uv         : TEXCOORD2;
                float  texLayer   : TEXCOORD3;   // texture array layer index
                float  fogFactor  : TEXCOORD4;
                UNITY_VERTEX_OUTPUT_STEREO
            };

            // ── Vertex shader ────────────────────────────────────────────
            Varyings vert(Attributes IN)
            {
                Varyings OUT;
                UNITY_SETUP_INSTANCE_ID(IN);
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(OUT);

                VertexPositionInputs posInputs = GetVertexPositionInputs(IN.positionOS);
                VertexNormalInputs   nrmInputs = GetVertexNormalInputs(IN.normalOS);

                float3 ws = posInputs.positionWS;
                ws.y -= PlanetCurvatureWorldDropY(ws);
                OUT.positionCS = TransformWorldToHClip(ws);
                OUT.positionWS = ws;
                OUT.normalWS   = nrmInputs.normalWS;
                OUT.uv         = IN.uv;
                OUT.texLayer   = IN.blockData.y;  // y = pre-computed tex array layer
                OUT.fogFactor  = ComputeFogFactor(OUT.positionCS.z);

                return OUT;
            }

            // ── Fragment shader ──────────────────────────────────────────
            half4 frag(Varyings IN) : SV_Target
            {
                // Sample the texture array using the layer baked per-vertex
                // Round texLayer to nearest int — float precision can drift
                float layer = round(IN.texLayer);
                half4 albedo = SAMPLE_TEXTURE2D_ARRAY(
                    _TextureArray, sampler_point_repeat_TextureArray,
                    IN.uv, layer);

                // Discard fully transparent pixels (for future transparent blocks)
                clip(albedo.a - 0.01);

                // ── URP PBR lighting ──────────────────────────────────────
                InputData    lightingInput  = (InputData)0;
                SurfaceData  surfaceData    = (SurfaceData)0;

                lightingInput.positionWS        = IN.positionWS;
                lightingInput.normalWS          = normalize(IN.normalWS);
                lightingInput.viewDirectionWS   = GetWorldSpaceNormalizeViewDir(IN.positionWS);
                lightingInput.shadowCoord       = TransformWorldToShadowCoord(IN.positionWS);
                lightingInput.fogCoord          = IN.fogFactor;
                lightingInput.bakedGI           = SampleSH(lightingInput.normalWS);
                lightingInput.normalizedScreenSpaceUV = GetNormalizedScreenSpaceUV(IN.positionCS);

                surfaceData.albedo      = albedo.rgb;
                surfaceData.alpha       = albedo.a;
                surfaceData.metallic    = _Metallic;
                surfaceData.smoothness  = _Smoothness;
                surfaceData.normalTS    = half3(0, 0, 1);
                surfaceData.occlusion   = 1.0;
                surfaceData.emission    = 0;

                half4 color = UniversalFragmentPBR(lightingInput, surfaceData);
                color.rgb   = MixFog(color.rgb, IN.fogFactor);
                return color;
            }

            ENDHLSL
        }

        // ── Shadow caster pass ───────────────────────────────────────────
        Pass
        {
            Name "ShadowCaster"
            Tags { "LightMode" = "ShadowCaster" }

            ZWrite On
            ZTest LEqual
            ColorMask 0
            Cull Back

            HLSLPROGRAM
            #pragma vertex   vertShadow
            #pragma fragment fragShadow

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Lighting.hlsl"
            #include "Include/PlanetCurvature.hlsl"

            CBUFFER_START(UnityPerMaterial)
                float _Smoothness;
                float _Metallic;
            CBUFFER_END

            struct AttributesShadow
            {
                float3 positionOS : POSITION;
                float3 normalOS   : NORMAL;
                float2 uv         : TEXCOORD0;
                float2 blockData  : TEXCOORD1;
            };

            struct VaryingsShadow
            {
                float4 positionCS : SV_POSITION;
            };

            VaryingsShadow vertShadow(AttributesShadow IN)
            {
                VaryingsShadow OUT;
                float3 posWS  = TransformObjectToWorld(IN.positionOS);
                posWS.y -= PlanetCurvatureWorldDropY(posWS);
                float3 normWS = TransformObjectToWorldNormal(IN.normalOS);
                Light  mainLight = GetMainLight();
                OUT.positionCS   = TransformWorldToHClip(
                    ApplyShadowBias(posWS, normWS, mainLight.direction));
                return OUT;
            }

            half4 fragShadow(VaryingsShadow IN) : SV_Target { return 0; }

            ENDHLSL
        }

        // ── Depth only pass (for depth prepass / SSAO) ───────────────────
        Pass
        {
            Name "DepthOnly"
            Tags { "LightMode" = "DepthOnly" }

            ZWrite On
            ColorMask R
            Cull Back

            HLSLPROGRAM
            #pragma vertex   vertDepth
            #pragma fragment fragDepth

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Include/PlanetCurvature.hlsl"

            struct AttributesDepth { float3 positionOS : POSITION; float2 uv : TEXCOORD0; float2 bd : TEXCOORD1; };
            struct VaryingsDepth   { float4 positionCS : SV_POSITION; };

            VaryingsDepth vertDepth(AttributesDepth IN)
            {
                VaryingsDepth OUT;
                float3 ws = TransformObjectToWorld(IN.positionOS);
                ws.y -= PlanetCurvatureWorldDropY(ws);
                OUT.positionCS = TransformWorldToHClip(ws);
                return OUT;
            }
            half fragDepth(VaryingsDepth IN) : SV_Target { return 0; }
            ENDHLSL
        }
    }
}
