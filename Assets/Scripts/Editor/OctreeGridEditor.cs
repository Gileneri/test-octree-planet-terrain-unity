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

    private bool presetsFoldout = true;
    private bool rebakeNeeded = false;

    public override void OnInspectorGUI()
    {
        serializedObject.Update();
        DrawPropertiesExcluding(serializedObject, "m_Script");

        EditorGUILayout.Space(4);
        DrawPresetsSection();

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
}
#endif