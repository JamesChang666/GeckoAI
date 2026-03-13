# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['src\\ai_labeller\\main.py'],
    pathex=['src'],
    binaries=[],
    datas=[('src/ai_labeller/assets', 'ai_labeller/assets'), ('src/ai_labeller/models', 'ai_labeller/models'), ('src/ai_labeller/train_runner.py', 'ai_labeller'), ('src/ai_labeller/build_training_runtime.py', 'ai_labeller'), ('src/ai_labeller/auto_build_training_runtime.py', 'ai_labeller')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='GeckoAI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='GeckoAI',
)
