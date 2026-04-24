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

    // Raios estimados para divisions=7, chunkResolution=8
    // Estrategia: niveis finos (lv0-3) ficam com mult BAIXO (perto do player)
    //             niveis grossos (lv4-6) tem mult ALTO (horizonte detalhado)
    // Isso evita a explosao cubica de nos que trava o jogo.
    private static readonly LodPreset[] Presets = new LodPreset[]
    {
        new LodPreset
        {
            name = "Performance",
            description = "Tudo perto do player. Minimo de nos. Para maquinas fracas ou debug.\n~130 nos totais  |  lv0: 16u  lv6: 768u",
            keys = new[]
            {
                new Keyframe(0f, 2f),
                new Keyframe(1f, 1.5f),
            }
        },
        new LodPreset
        {
            name = "Padrao",
            description = "Balanceado. Niveis finos perto, niveis grossos um pouco mais longe.\n~500 nos totais  |  lv0: 16u  lv6: 1024u",
            keys = new[]
            {
                new Keyframe(0f, 2f),
                new Keyframe(0.5f, 2f),
                new Keyframe(1f, 2f),
            }
        },
        new LodPreset
        {
            name = "Horizonte Suave",
            description = "Finos perto (mult=2), grossos esticados ate 5x. Bom visual sem custo excessivo.\n~1800 nos totais  |  lv0: 16u  lv6: 2560u",
            keys = new[]
            {
                new Keyframe(0f, 2f),
                new Keyframe(0.55f, 2f),
                new Keyframe(1f, 5f),
            }
        },
        new LodPreset
        {
            name = "Horizonte Detalhado",
            description = "Finos perto, grossos bem esticados. Horizonte notavelmente mais rico.\n~3400 nos totais  |  lv0: 16u  lv6: 4096u",
            keys = new[]
            {
                new Keyframe(0f, 2f),
                new Keyframe(0.5f, 2f),
                new Keyframe(0.7f, 4f),
                new Keyframe(1f, 8f),
            }
        },
        new LodPreset
        {
            name = "Horizonte Maximo",
            description = "Finos perto, grossos extremamente longe. Pesado — aumente maxJobsPerFrame para 300+.\n~8000 nos totais  |  lv0: 16u  lv6: 7168u",
            keys = new[]
            {
                new Keyframe(0f, 2f),
                new Keyframe(0.5f, 2f),
                new Keyframe(0.65f, 5f),
                new Keyframe(1f, 14f),
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
                "DICA: a curva deve ser PLANA nos niveis finos (X=0 a 0.5) e\n" +
                "subir apenas nos niveis grossos (X=0.5 a 1.0) para evitar\n" +
                "explosao de nos e travamentos.",
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