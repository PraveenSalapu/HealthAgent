"""
Script to copy only essential files to a clean directory for git repository.
Run this to create a clean version ready for pushing to new repo.
"""

import os
import shutil
from pathlib import Path

# Essential files and directories to copy
ESSENTIAL_FILES = [
    # Main app
    "app_modular.py",
    "requirements.txt",
    ".gitignore",

    # Streamlit config
    ".streamlit/secrets.toml.example",

    # Agents module
    "agents/__init__.py",
    "agents/base_agent.py",
    "agents/gemini_agent.py",
    "agents/lightweight_rag_agent.py",
    "agents/agent_manager.py",

    # Config module
    "config/__init__.py",
    "config/settings.py",
    "config/document_metadata.py",

    # Models module
    "models/__init__.py",
    "models/model_loader.py",
    "models/predictor.py",

    # UI module
    "ui/__init__.py",
    "ui/forms.py",
    "ui/visualizations.py",
    "ui/enhanced_visualizations.py",
    "ui/chat_interface.py",
    "ui/styles.py",

    # Utils module
    "utils/__init__.py",
    "utils/helpers.py",
    "utils/lightweight_rag.py",

    # Model files
    "model_output/xgb_model.json",
    "model_output/preprocessing_config.json",
    "model_output/optimal_threshold.json",
    "model_output/diabetic_averages.json",

    # Pages
    "pages/1_Admin_Document_Upload.py",
    
    # Documentation
    "README.md",
]

def copy_essential_files(source_dir="."):
    """Copy essential files to a new clean directory."""

    # Create clean directory
    clean_dir = Path(source_dir).parent / "HealthAgentDiabetic-Clean"

    if clean_dir.exists():
        print(f"⚠️  Directory {clean_dir} already exists!")
        # Force delete and recreate
        shutil.rmtree(clean_dir)

    clean_dir.mkdir()
    print(f"✅ Created clean directory: {clean_dir}")

    # Copy each essential file
    copied_count = 0
    missing_files = []

    for file_path in ESSENTIAL_FILES:
        source_file = Path(source_dir) / file_path
        dest_file = clean_dir / file_path

        if source_file.exists():
            # Create parent directories if needed
            dest_file.parent.mkdir(parents=True, exist_ok=True)

            # Copy file
            shutil.copy2(source_file, dest_file)
            print(f"  ✅ Copied: {file_path}")
            copied_count += 1
        else:
            print(f"  ⚠️  Missing: {file_path}")
            missing_files.append(file_path)

    # Summary
    print("\n" + "="*60)
    print(f"✅ Copied {copied_count}/{len(ESSENTIAL_FILES)} files")

    if missing_files:
        print(f"\n⚠️  Missing {len(missing_files)} files:")
        for f in missing_files:
            print(f"   - {f}")

    print(f"\n📁 Clean directory created at: {clean_dir}")
    print("\n🚀 Next steps:")
    print(f"   1. cd {clean_dir}")
    print("   2. git init")
    print("   3. git add .")
    print('   4. git commit -m "Initial commit: Clean production application"')
    print("   5. git remote add origin <your-repo-url>")
    print("   6. git push -u origin main")
    print("="*60)

if __name__ == "__main__":
    print("="*60)
    print("🧹 Essential Files Copy Script (Auto-Run)")
    print("="*60)
    copy_essential_files()
