#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;

[CustomEditor(typeof(OctreeGrid))]
public class OctreeGridEditor : Editor
{
    private struct LodPreset
    {
        public string name;
        public string description;
        public Keyframe[] keys;
    }

    private struct FarDistancePreset
    {
        public string name;
        public string description;
        public bool enabled;
        public float gridSize;
        public int gridResolution;
        public int texResolution;
        public float refreshSeconds;
        public float nearStart;
        public float nearEnd;
        public float farStart;
        public float farEnd;
        public float fogHorizonMix;
        public float backgroundClipEpsilon;
        public float playerHoleInner;
        public float playerHoleOuter;
        public bool deepUndergroundFade;
        public float undergroundFadeStart;
        public float undergroundFadeEnd;
        public bool periodicRefresh;
        public float underlayYOffset;
    }

    // Raios estimados para divisions=7, chunkResolution=8.
    // Esta versao prioriza testes de "fino um pouco mais longe" sem alterar
    // a arquitetura: mexe apenas na lodDistanceCurve.
    //
    // Estrategia:
    // - Mantem lv4-6 com crescimento progressivo (horizonte detalhado).
    // - Sobe levemente lv0-3 para empurrar detalhe fino para mais longe.
    // - Evita saltos bruscos perto de X=0.5 para reduzir churn de subdivisao.
    private static readonly LodPreset[] Presets = new LodPreset[]
    {
        new LodPreset
        {
            name = "Performance Atual",
            description = "Base conservadora para FPS alto. Bom para comparar com presets mais finos.\n~130 nos totais  |  lv0: 16u  lv6: 768u",
            keys = new[]
            {
                new Keyframe(0f, 2f),
                new Keyframe(1f, 1.5f),
            }
        },
        new LodPreset
        {
            name = "Padrao Atual",
            description = "Referencia balanceada do projeto. Mantem finos mais perto e custo previsivel.\n~500 nos totais  |  lv0: 16u  lv6: 1024u",
            keys = new[]
            {
                new Keyframe(0f, 2f),
                new Keyframe(0.5f, 2f),
                new Keyframe(1f, 2f),
            }
        },
        new LodPreset
        {
            name = "Fino+ Curto Alcance",
            description = "Primeiro passo para detalhe fino mais longe: aumenta discretamente lv0-3.\n~800-1200 nos  |  lv0: 19u  lv3: 166u  lv6: 1536u",
            keys = new[]
            {
                new Keyframe(0f, 2.4f),
                new Keyframe(0.35f, 2.3f),
                new Keyframe(0.6f, 2.8f),
                new Keyframe(1f, 3f),
            }
        },
        new LodPreset
        {
            name = "Fino+ Medio Alcance",
            description = "Detalhe fino claramente mais distante, mantendo transicao suave para lv4-6.\n~1600-2400 nos  |  lv0: 22u  lv3: 205u  lv6: 2304u",
            keys = new[]
            {
                new Keyframe(0f, 2.8f),
                new Keyframe(0.35f, 2.7f),
                new Keyframe(0.6f, 3.3f),
                new Keyframe(0.8f, 4f),
                new Keyframe(1f, 4.5f),
            }
        },
        new LodPreset
        {
            name = "Fino+ Longo Alcance",
            description = "Teste agressivo para validar limite do hardware: fino longe e horizonte denso.\n~2800-4200 nos  |  lv0: 26u  lv3: 243u  lv6: 3072u",
            keys = new[]
            {
                new Keyframe(0f, 3.2f),
                new Keyframe(0.35f, 3.1f),
                new Keyframe(0.6f, 3.8f),
                new Keyframe(0.8f, 5f),
                new Keyframe(1f, 6f),
            }
        },
        new LodPreset
        {
            name = "Fino+ Estresse",
            description = "Stress test para encontrar teto de desempenho com detalhe fino distante.\n~4500+ nos  |  lv0: 29u  lv3: 282u  lv6: 4096u",
            keys = new[]
            {
                new Keyframe(0f, 3.6f),
                new Keyframe(0.35f, 3.5f),
                new Keyframe(0.6f, 4.4f),
                new Keyframe(0.8f, 5.8f),
                new Keyframe(1f, 8f),
            }
        },
    };

    private static readonly FarDistancePreset[] FarPresets = new FarDistancePreset[]
    {
        new FarDistancePreset
        {
            name = "Far Off",
            description = "Desliga o far distance heightmap para baseline puro da octree.",
            enabled = false,
            gridSize = 10000f,
            gridResolution = 128,
            texResolution = 512,
            refreshSeconds = 1.0f,
            nearStart = 999999f,
            nearEnd = 1000000f,
            farStart = 1000001f,
            farEnd = 1000002f,
            fogHorizonMix = 0f,
            backgroundClipEpsilon = 1e-5f,
            playerHoleInner = 400f,
            playerHoleOuter = 3600f,
            deepUndergroundFade = false,
            undergroundFadeStart = 32f,
            undergroundFadeEnd = 400f,
            periodicRefresh = false,
            underlayYOffset = 0f,
        },
        new FarDistancePreset
        {
            name = "Far Safe",
            description = "Preset inicial estavel: transicao longa e custo moderado.",
            enabled = true,
            gridSize = 50000f,
            gridResolution = 128,
            texResolution = 512,
            refreshSeconds = 2.0f,
            nearStart = 1800f,
            nearEnd = 3600f,
            farStart = 20000f,
            farEnd = 28000f,
            fogHorizonMix = 0.55f,
            backgroundClipEpsilon = 1e-5f,
            playerHoleInner = 400f,
            playerHoleOuter = 3600f,
            deepUndergroundFade = false,
            undergroundFadeStart = 32f,
            undergroundFadeEnd = 400f,
            periodicRefresh = false,
            underlayYOffset = 0f,
        },
        new FarDistancePreset
        {
            name = "Far Quality",
            description = "Visual mais suave no horizonte (custo GPU maior).",
            enabled = true,
            gridSize = 50000f,
            gridResolution = 192,
            texResolution = 1024,
            refreshSeconds = 1.5f,
            nearStart = 2200f,
            nearEnd = 4200f,
            farStart = 26000f,
            farEnd = 34000f,
            fogHorizonMix = 0.65f,
            backgroundClipEpsilon = 8e-6f,
            playerHoleInner = 500f,
            playerHoleOuter = 4200f,
            deepUndergroundFade = false,
            undergroundFadeStart = 32f,
            undergroundFadeEnd = 400f,
            periodicRefresh = false,
            underlayYOffset = 0f,
        },
        new FarDistancePreset
        {
            name = "Far Performance",
            description = "Prioriza FPS: malha/textura menores e refresh mais espaçado.",
            enabled = true,
            gridSize = 50000f,
            gridResolution = 128,
            texResolution = 512,
            refreshSeconds = 3.0f,
            nearStart = 1600f,
            nearEnd = 3000f,
            farStart = 18000f,
            farEnd = 24000f,
            fogHorizonMix = 0.45f,
            backgroundClipEpsilon = 1.2e-5f,
            playerHoleInner = 350f,
            playerHoleOuter = 3200f,
            deepUndergroundFade = false,
            undergroundFadeStart = 32f,
            undergroundFadeEnd = 400f,
            periodicRefresh = false,
            underlayYOffset = 0f,
        },
    };

    private bool presetsFoldout = true;
    private bool farPresetsFoldout = true;
    private bool rebakeNeeded = false;

    public override void OnInspectorGUI()
    {
        serializedObject.Update();
        DrawPropertiesExcluding(serializedObject, "m_Script");

        EditorGUILayout.Space(4);
        DrawPresetsSection();
        EditorGUILayout.Space(4);
        DrawFarPresetsSection();

        if (rebakeNeeded)
            DrawRebakeButton();

        serializedObject.ApplyModifiedProperties();
    }

    private void DrawPresetsSection()
    {
        presetsFoldout = EditorGUILayout.BeginFoldoutHeaderGroup(presetsFoldout, "Presets de LOD");
        if (presetsFoldout)
        {
            EditorGUILayout.HelpBox(
                "Clique em Aplicar para sobrescrever a lodDistanceCurve acima.\n" +
                "Os raios e contagens assumem divisions=7 e chunkResolution=8.\n" +
                "Depois clique em Rebake & Apply para ver o efeito sem reiniciar.\n\n" +
                "Nesta versao os presets Fino+ aumentam gradualmente X=0..0.6\n" +
                "para levar detalhe fino mais longe sem alterar a estrutura.\n" +
                "Teste em ordem: Curto -> Medio -> Longo -> Estresse.",
                MessageType.Info);

            EditorGUILayout.Space(2);

            foreach (var preset in Presets)
            {
                EditorGUILayout.BeginVertical(EditorStyles.helpBox);
                EditorGUILayout.BeginHorizontal();
                GUILayout.Label(preset.name, EditorStyles.boldLabel);
                if (GUILayout.Button("Aplicar", GUILayout.Width(70)))
                {
                    ApplyPreset(preset);
                    rebakeNeeded = true;
                }
                EditorGUILayout.EndHorizontal();

                var style = new GUIStyle(EditorStyles.miniLabel)
                { wordWrap = true, normal = { textColor = new Color(0.6f, 0.6f, 0.6f) } };
                GUILayout.Label(preset.description, style);

                EditorGUILayout.EndVertical();
                EditorGUILayout.Space(2);
            }
        }
        EditorGUILayout.EndFoldoutHeaderGroup();
    }

    private void DrawRebakeButton()
    {
        EditorGUILayout.Space(4);
        GUI.backgroundColor = new Color(0.4f, 0.9f, 0.4f);
        if (GUILayout.Button("Rebake & Apply  (apenas em runtime)", GUILayout.Height(30)))
        {
            var grid = (OctreeGrid)target;
            if (Application.isPlaying)
            {
                grid.RebakeLodRadii();
                rebakeNeeded = false;
                Debug.Log("[OctreeGridEditor] LOD radii rebaked e aplicados.");
            }
            else
            {
                Debug.LogWarning("[OctreeGridEditor] Entre em Play Mode para aplicar o rebake.");
            }
        }
        GUI.backgroundColor = Color.white;
    }

    private void DrawFarPresetsSection()
    {
        farPresetsFoldout = EditorGUILayout.BeginFoldoutHeaderGroup(farPresetsFoldout, "Presets Far Distance Heightmap");
        if (farPresetsFoldout)
        {
            EditorGUILayout.HelpBox(
                "Presets iniciais para validar custo/beneficio do horizonte distante.\n" +
                "Fluxo recomendado: Far Off -> Far Safe -> Far Quality/Performance.\n" +
                "Use o Profiler para comparar batches, GPU frame time e estabilidade visual.",
                MessageType.Info);

            foreach (var preset in FarPresets)
            {
                EditorGUILayout.BeginVertical(EditorStyles.helpBox);
                EditorGUILayout.BeginHorizontal();
                GUILayout.Label(preset.name, EditorStyles.boldLabel);
                if (GUILayout.Button("Aplicar", GUILayout.Width(70)))
                    ApplyFarPreset(preset);
                EditorGUILayout.EndHorizontal();

                var style = new GUIStyle(EditorStyles.miniLabel)
                { wordWrap = true, normal = { textColor = new Color(0.6f, 0.6f, 0.6f) } };
                GUILayout.Label(preset.description, style);
                EditorGUILayout.EndVertical();
                EditorGUILayout.Space(2);
            }
        }
        EditorGUILayout.EndFoldoutHeaderGroup();
    }

    private void ApplyPreset(LodPreset preset)
    {
        var grid = (OctreeGrid)target;
        Undo.RecordObject(grid, "LOD Preset: " + preset.name);

        var curve = new AnimationCurve(preset.keys);
        for (int i = 0; i < curve.length; i++)
            AnimationUtility.SetKeyLeftTangentMode(curve, i, AnimationUtility.TangentMode.Auto);

        grid.lodDistanceCurve = curve;
        EditorUtility.SetDirty(grid);
    }

    private void ApplyFarPreset(FarDistancePreset preset)
    {
        var grid = (OctreeGrid)target;
        Undo.RecordObject(grid, "Far Preset: " + preset.name);

        grid.enableFarDistanceHeightmap = preset.enabled;
        grid.farDistanceGridWorldSize = preset.gridSize;
        grid.farDistanceGridResolution = preset.gridResolution;
        grid.farDistanceHeightTextureResolution = preset.texResolution;
        grid.farDistanceHeightRefreshSeconds = preset.refreshSeconds;
        grid.farDistanceFadeStart = preset.nearStart;
        grid.farDistanceFadeEnd = preset.nearEnd;
        grid.farDistanceFarFadeStart = preset.farStart;
        grid.farDistanceFarFadeEnd = preset.farEnd;
        grid.farDistanceFogHorizonMix = preset.fogHorizonMix;
        grid.farDistanceBackgroundClipEpsilon = preset.backgroundClipEpsilon;
        grid.farDistancePlayerHoleInner = preset.playerHoleInner;
        grid.farDistancePlayerHoleOuter = preset.playerHoleOuter;
        grid.farDistanceDeepUndergroundFarFade = preset.deepUndergroundFade;
        grid.farDistanceUndergroundFadeStart = preset.undergroundFadeStart;
        grid.farDistanceUndergroundFadeEnd = preset.undergroundFadeEnd;
        grid.farDistancePeriodicRefresh = preset.periodicRefresh;
        grid.farDistanceUnderlayYOffset = preset.underlayYOffset;
        EditorUtility.SetDirty(grid);
    }
}
#endif