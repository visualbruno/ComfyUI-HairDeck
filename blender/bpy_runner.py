"""Blender's command line on the bpy module: `python bpy_runner.py -b --python SCRIPT -- ARGS` runs SCRIPT as
`blender -b --python SCRIPT -- ARGS` would (sys.argv[0] the script, ARGS after '--'), so HairDeck's Blender scripts
and their callers work unchanged with bpy in place of a Blender install. Other Blender options are ignored.

Environment (set by hairdeck_env.blender_env before the bpy import, which reads them):
  BLENDER_USER_*            Blender's user folders (the extensions repository the VRM Add-on goes into)
  HAIRDECK_VRM_ADDON        module of the VRM Add-on for Blender (bl_ext.user_default.vrm, unpacked beforehand by
                            hairdeck_env.install_vrm_addon): enabled and patched before the script runs (Blender 5.2
                            glTF material export; MToon conversion in background mode)
With a Blender executable the scripts run directly, without this runner.
"""
import os
import runpy
import sys

def enable_vrm(module):
    """Enable the VRM Add-on (unpacked beforehand by hairdeck_env.install_vrm_addon) and patch it for Blender 5.2."""
    import addon_utils
    import bpy

    addon_utils.enable(module, default_set=True)
    # bpy.ops makes a wrapper for any name, so hasattr() cannot tell; dir() lists the registered operators only
    if 'vrm' not in dir(bpy.ops.import_scene) or 'vrm' not in dir(bpy.ops.export_scene):
        raise RuntimeError(f'the VRM add-on ({module}) did not register; see the messages above')
    patch_vrm_gltf_material(module)
    patch_vrm_mtoon_convert(module)


def patch_vrm_gltf_material(module):
    """VRM Add-on 4.10.x unpacks two values from io_scene_gltf2's gather_material; the glTF exporter of Blender 5.2
    returns three (blender material, glTF material, info), so every material fails to export. Rebind the add-on's
    wrapper in each module that imported it; the glTF exporter itself is left alone."""
    def gather_gltf2_io_material(material, export_settings):
        from io_scene_gltf2.blender.exp.material.materials import gather_material
        r = gather_material(material, export_settings)
        return r[1] if len(r) == 3 else r[0]

    for name, mod in list(sys.modules.items()):
        if name.startswith(module) and hasattr(mod, 'gather_gltf2_io_material'):
            mod.gather_gltf2_io_material = gather_gltf2_io_material


def patch_vrm_mtoon_convert(module):
    """bpy.ops.vrm.convert_material_to_mtoon1 builds the MToon node tree but leaves mtoon1.enabled unset: the add-on's
    scene watcher sets it later from the UI event loop, which never runs in a background script. The VRM 0.x export
    then writes the material as VRM_USE_GLTFSHADER (plain glTF) instead of VRM/MToon. Make the operator's conversion
    also enable the flag, as the watcher would."""
    ops = sys.modules.get(module + '.editor.mtoon1.ops')
    if ops is None or not hasattr(ops, 'convert_material_to_mtoon1'):
        return
    original, get_extension = ops.convert_material_to_mtoon1, ops.get_material_extension

    def convert_material_to_mtoon1(material, *args, **kwargs):
        original(material, *args, **kwargs)
        get_extension(material).mtoon1.enabled = True

    ops.convert_material_to_mtoon1 = convert_material_to_mtoon1


def main():
    argv = sys.argv[1:]
    script = argv[argv.index('--python') + 1]
    rest = argv[argv.index('--'):] if '--' in argv else []
    import bpy  # noqa: F401  (after the BLENDER_USER_* environment is in place)

    if os.environ.get('HAIRDECK_VRM_ADDON'):
        enable_vrm(os.environ['HAIRDECK_VRM_ADDON'])
    sys.argv = [script] + rest
    runpy.run_path(script, run_name='__main__')


if __name__ == '__main__':
    main()
