# setup_requirements.py
import os
import sys
import subprocess

print("🔧 Analizador de dependencias - IA SNN Project\n")

# Detectar si estamos en venv
in_venv = sys.prefix != sys.base_prefix
venv_name = os.path.basename(sys.prefix)

print(f"📍 Ruta actual: {os.getcwd()}")
print(f"🐍 Usando Python: {sys.executable}")
print(f"🛡️  Entorno virtual: {'SÍ (' + venv_name + ')' if in_venv else 'NO'}")

# Paquetes necesarios para tu proyecto
required_packages = [
    "numpy",
    "matplotlib",
    "opencv-python",      # para screen_interface (cv2)
    "pillow",             # por si acaso
    "scipy",              # útil para cálculos
]

print("\n📦 Paquetes que se van a instalar:")
for pkg in required_packages:
    print(f"   • {pkg}")

# Crear requirements.txt
with open("requirements.txt", "w", encoding="utf-8") as f:
    for pkg in required_packages:
        f.write(pkg + "\n")

print(f"\n✅ Archivo requirements.txt creado correctamente.")

# Instalar paquetes
print("\n🚀 Instalando dependencias...")
try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "--upgrade"])
    print("\n🎉 ¡Todas las dependencias se instalaron correctamente!")
except Exception as e:
    print(f"\n❌ Error durante la instalación: {e}")

# Verificar instalación
print("\n📋 Verificación de paquetes instalados:")
for pkg in required_packages:
    try:
        __import__(pkg.replace("-", "_").split("[")[0])
        print(f"   ✅ {pkg} → OK")
    except ImportError:
        print(f"   ❌ {pkg} → No encontrado")

print("\n" + "="*60)
print("✅ Setup finalizado.")
print("Ahora puedes ejecutar:")
print("   python visualizer.py")
print("   python main.py")
print("="*60)