bl_info = {
    "name": "Knight Online Character Importer",
    "author": "Your Name",
    "version": (1, 0, 0), # Add minor version
    "blender": (4, 2, 0),
    "location": "File > Import > Knight Online Character (.n3chr)",
    "description": "Import Knight Online Character (.n3chr) files",
    "warning": "",
    "doc_url": "",
    "category": "Import-Export",
}

import bpy
import os
import sys
import subprocess
from bpy.types import Operator, Panel, AddonPreferences
from bpy.props import StringProperty, BoolProperty
from bpy_extras.io_utils import ImportHelper
from bpy.types import Operator
import sys
import os

class KnightOnlinePreferences(AddonPreferences):
    bl_idname = __package__

    def draw(self, context):
        layout = self.layout
        
        # Check if dependencies are installed correctly
        try:
            # First try importing from packages directory
            addon_dir = os.path.dirname(os.path.realpath(__file__))
            packages_dir = os.path.join(addon_dir, "packages")
            if packages_dir not in sys.path:
                sys.path.append(packages_dir)
            
            # Try importing PIL from packages first
            from packages import PIL
            from packages.PIL import Image
            pil_installed = True
        except ImportError:
            try:
                # Try system-wide PIL as fallback
                from PIL import Image
                pil_installed = True
            except ImportError:
                pil_installed = False
            
        try:
            import cv2
            opencv_installed = True
        except ImportError:
            opencv_installed = False

        # PIL installation section
        box = layout.box()
        row = box.row()
        if not pil_installed:
            row.operator(InstallDependenciesOperator.bl_idname, text="Install PIL").package = "Pillow"
            row.label(text="PIL: Not Installed", icon='ERROR')
        else:
            row.label(text="PIL: Installed", icon='CHECKMARK')

        # OpenCV installation section    
        box = layout.box()
        row = box.row()
        if not opencv_installed:
            row.operator(InstallDependenciesOperator.bl_idname, text="Install OpenCV").package = "opencv-python"
            row.label(text="OpenCV: Not Installed", icon='ERROR')
        else:
            row.label(text="OpenCV: Installed", icon='CHECKMARK')

# Add packages directory to Python path
addon_dir = os.path.dirname(os.path.realpath(__file__))
if (addon_dir not in sys.path):
    sys.path.append(addon_dir)

class InstallDependenciesOperator(Operator):
    """Install required dependencies"""
    bl_idname = "wm.install_ko_dependencies"
    bl_label = "Install Dependencies"
    
    package: StringProperty()
    
    def execute(self, context):
        addon_dir = os.path.dirname(os.path.realpath(__file__))
        packages_dir = os.path.join(addon_dir, "packages")
        os.makedirs(packages_dir, exist_ok=True)
        
        python_exe = sys.executable
        
        try:
            # First ensure pip is available
            subprocess.check_call([
                python_exe, 
                "-m", "ensurepip"
            ])
            
            # Install the package
            subprocess.check_call([
                python_exe,
                "-m", "pip",
                "install",
                "--target", packages_dir,
                self.package,
                "--no-deps"  # Avoid dependency conflicts
            ])
            
            # Force reload sys.modules to recognize new packages
            import importlib
            importlib.invalidate_caches()
            
            # Add packages dir to path if not already there
            if packages_dir not in sys.path:
                sys.path.append(packages_dir)
            
            self.report({'INFO'}, f"Successfully installed {self.package}")
            return {'FINISHED'}
            
        except subprocess.CalledProcessError as e:
            self.report({'ERROR'}, f"Failed to install {self.package}: {str(e)}")
            return {'CANCELLED'}

class DependenciesPanel(Panel):
    """Dependencies installer panel"""
    bl_label = "KO Character Dependencies"
    bl_idname = "VIEW3D_PT_ko_dependencies"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'KO Import'
    
    def draw(self, context):
        layout = self.layout
        
        # Check if dependencies are installed
        try:
            from packages import PIL
            pil_installed = True
        except ImportError:
            pil_installed = False
            
        try:
            from packages import cv2
            opencv_installed = True
        except ImportError:
            opencv_installed = False
        
        # PIL installation button
        row = layout.row()
        if not pil_installed:
            row.operator(InstallDependenciesOperator.bl_idname, text="Install PIL").package = "Pillow"
        else:
            row.label(text="PIL: Installed ✓")
            
        # OpenCV installation button    
        row = layout.row()
        if not opencv_installed:
            row.operator(InstallDependenciesOperator.bl_idname, text="Install OpenCV").package = "opencv-python"
        else:
            row.label(text="OpenCV: Installed ✓")

import ctypes
from pathlib import Path
from enum import Enum
import struct
from uuid import uuid4
from typing import Annotated
from io import BytesIO
import tempfile
import os
import ensurepip
import pip
from copy import deepcopy

import bpy
from bpy_extras.io_utils import ImportHelper
from bpy.props import StringProperty, BoolProperty, EnumProperty
from bpy.types import Operator
import mathutils
from math import radians

# Remove the immediate dependency check and make it conditional
def check_dependencies():
    """Check if required dependencies are available"""
    try:
        # Try packages directory first
        addon_dir = os.path.dirname(os.path.realpath(__file__))
        packages_dir = os.path.join(addon_dir, "packages")
        if packages_dir not in sys.path:
            sys.path.append(packages_dir)
            
        try:
            from packages import PIL
            from packages.PIL import Image
            USE_PIL = True
        except ImportError:
            try:
                from PIL import Image
                USE_PIL = True
            except ImportError:
                USE_PIL = False
    except:
        USE_PIL = False
    
    try:
        import cv2
        USE_OPENCV = True
    except ImportError:
        USE_OPENCV = False
    
    return USE_PIL or USE_OPENCV

def load_texture(buffer, width, height):
    try:
        # Try packages directory first
        from packages import PIL
        from packages.PIL import Image
        return Image.frombuffer('RGBA', (width, height), buffer, 'raw', 'RGBA', 0, 1)
    except ImportError:
        try:
            # Try system PIL as fallback
            from PIL import Image
            return Image.frombuffer('RGBA', (width, height), buffer, 'raw', 'RGBA', 0, 1)
        except ImportError:
            try:
                # Try OpenCV as last resort
                import cv2
                import numpy as np
                img_array = np.frombuffer(buffer, dtype=np.uint8)
                img_array = img_array.reshape((height, width, 4))
                return cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGRA)
            except ImportError:
                raise ImportError("Please install either PIL or OpenCV from the addon preferences")

MAX_CHR_LOD = 4
MAX_CHR_ANI_PART = 2

# Transforms for going from right-handed to left-handed coordinates
map_mtx = mathutils.Matrix(((1, 0, 0, 0), (0, 0, -1, 0), (0, 1, 0, 0), (0, 0, 0, 1)))
map_mtx = map_mtx @ mathutils.Matrix(((-1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)))

# =================================

# The code below is from here and is used to decompress the DXT textures:
# > https://github.com/leamsii/Python-DXT-Decompress

"""
S3TC DXT1/DXT5 Texture Decompression

Inspired by Benjamin Dobell

Original C++ code https://github.com/Benjamin-Dobell/s3tc-dxt-decompression
"""

def unpack(_bytes):
    STRUCT_SIGNS = {
    1 : 'B',
    2 : 'H',
    4 : 'I',
    8 : 'Q'
    }
    return struct.unpack('<' + STRUCT_SIGNS[len(_bytes)], _bytes)[0]

# This function converts RGB565 format to raw pixels
def unpackRGB(packed):
    R = (packed >> 11) & 0x1F
    G = (packed >> 5) & 0x3F
    B = (packed) & 0x1F

    R = (R << 3) | (R >> 2)
    G = (G << 2) | (G >> 4)
    B = (B << 3) | (B >> 2)

    return (R, G, B, 255)

class DXTBuffer:
    def __init__(self, width, height):
        self.width = width
        self.height = height

        self.block_countx = self.width // 4
        self.block_county = self.height// 4

        self.decompressed_buffer = ["X"] * ((width * height) * 2) # Dont ask me why
        print(f"Log: New DXTBuffer instance created {width}x{height}")


    def DXT5Decompress(self, file):
        print("Log: DTX5 Decompressing..")

        # Loop through each block and decompress it
        for row in range(self.block_countx):
            for col in range(self.block_county):
                
                # Get the alpha values
                a0 = unpack(file.read(1))
                a1 = unpack(file.read(1))
                atable = file.read(6)

                acode0 = atable[2] | (atable[3] << 8) | (atable[4] << 16) | (atable[5] << 24)
                acode1 = atable[0] | (atable[1] << 8)

                # Color 1 color 2, color look up table
                c0 = unpack(file.read(2))
                c1 = unpack(file.read(2))
                ctable = unpack(file.read(4))

                # The 4x4 Lookup table loop
                for j in range(4):
                    for i in range(4):
                        alpha = self.getAlpha(j, i, a0, a1, atable, acode0, acode1)
                        self.getColors(row * 4, col * 4, i, j, ctable, unpackRGB(c0) ,unpackRGB(c1), alpha) # Set the color for the current pixel

        print("Log: DXT Buffer decompressed and returned successfully.")
        return b''.join([_ for _ in self.decompressed_buffer if _ != 'X'])


    def DXT1Decompress(self, file):
        print(f"Log: DTX1 Decompressing..")
        # Loop through each block and decompress it
        for row in range(self.block_countx):
            for col in range(self.block_county):

                # Color 1 color 2, color look up table
                c0 = unpack(file.read(2))
                c1 = unpack(file.read(2))
                ctable = unpack(file.read(4))

                # The 4x4 Lookup table loop
                for j in range(4):
                    for i in range(4):
                        self.getColors(row * 4, col * 4, i, j, ctable, unpackRGB(c0) ,unpackRGB(c1), 255) # Set the color for the current pixel

        print("Log: DXT Buffer decompressed and returned successfully.")
        return b''.join([_ for _ in self.decompressed_buffer if _ != 'X'])

    def getColors(self, x, y, i, j, ctable, c0, c1, alpha):
        code = (ctable >> ( 2 * (4 * i + j))) & 0x03 # Get the color of the current pixel
        pixel_color = None

        r0 = c0[0]
        g0 = c0[1]
        b0 = c0[2]

        r1 = c1[0]
        g1 = c1[1]
        b1 = c1[2]

        # Main two colors
        if code == 0:
            pixel_color = (r0, g0, b0, alpha)
        if code == 1:
            pixel_color = (r1, g1, b1, alpha)

        # Use the lookup table to determine the other two colors
        if c0 > c1:
            if code == 2:
                pixel_color = ((2*r0+r1)//3, (2*g0+g1)//3, (2*b0+b1)//3, alpha)
            if code == 3:
                pixel_color = ((r0+2*r1)//3, (g0+2*g1)//3, (b0+2*b1)//3, alpha)
        else:
            if code == 2:
                pixel_color = ((r0+r1)//2, (g0+g1)//2, (b0+b1)//2, alpha)
            if code == 3:
                pixel_color = (0, 0, 0, alpha)

        # While not surpassing the image dimensions, assign pixels the colors
        if (x + i) < self.width:
            self.decompressed_buffer[(y + j) * self.width + (x + i)] = struct.pack('<B', pixel_color[0]) + \
                struct.pack('<B', pixel_color[1]) + struct.pack('<B', pixel_color[2]) + struct.pack('<B', pixel_color[3])

    def getAlpha(self, i, j, a0, a1, atable, acode0, acode1):

        # Using the same method as the colors calculate the alpha values

        alpha = 255
        alpha_index = 3 * (4 * j+i)
        alpha_code = None

        if alpha_index <= 12:
            alpha_code = (acode1 >> alpha_index) & 0x07
        elif alpha_index == 15:
            alpha_code = (acode1 >> 15) | ((acode0 << 1) & 0x06)
        else:
            alpha_code = (acode0 >> (alpha_index - 16)) & 0x07

        if alpha_code == 0:
            alpha = a0
        elif alpha_code == 1:
            alpha = a1
        else:
            if a0 > a1:
                alpha = ((8-alpha_code) * a0 + (alpha_code-1) * a1) // 7
            else:
                if alpha_code == 6:
                    alpha = 0
                elif alpha_code == 7:
                    alpha = 255
                elif alpha_code == 5:
                    alpha = (1 * a0 + 4 * a1) // 5
                elif alpha_code == 4:
                    alpha = (2 * a0 + 3 * a1) // 5
                elif alpha_code == 3:
                    alpha = (3 * a0 + 2 * a1) // 5
                elif alpha_code == 2:
                    alpha = (4 * a0 + 1 * a1) // 5
                else:
                    alpha = 0 # For safety
        return alpha

# =================================

# ========== START TYPES ==========

class LODCtrlValue(ctypes.Structure):
    _fields_ = [
        ("fDist", ctypes.c_float),
        ("iNumVertices", ctypes.c_int)
    ]

class Matrix44(ctypes.Structure):
    _fields_ = [
        ("_11", ctypes.c_float), ("_12", ctypes.c_float), ("_13", ctypes.c_float), ("_14", ctypes.c_float),
        ("_21", ctypes.c_float), ("_22", ctypes.c_float), ("_23", ctypes.c_float), ("_24", ctypes.c_float),
        ("_31", ctypes.c_float), ("_32", ctypes.c_float), ("_33", ctypes.c_float), ("_34", ctypes.c_float),
        ("_41", ctypes.c_float), ("_42", ctypes.c_float), ("_43", ctypes.c_float), ("_44", ctypes.c_float)
    ]

class UVVector(ctypes.Structure):
    _fields_ = [
        ("v", ctypes.c_float),
        ("u", ctypes.c_float)
    ]

class Vector(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("z", ctypes.c_float)
    ]

class Vertex(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("z", ctypes.c_float),
        ("n", Vector)
    ]

class VertexWithUV(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("z", ctypes.c_float),
        ("n", Vector),
        ("v", ctypes.c_float),
        ("u", ctypes.c_float)
    ]

class VertexSkinned(ctypes.Structure):
    _fields_ = [
        ("vOrigin", Vector),
        ("nAffect", ctypes.c_int),
        ("pnJoints", ctypes.c_int),  # Pointer
        ("pfWeights", ctypes.c_int)  # Pointer
    ]

class Quaternion(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("z", ctypes.c_float),
        ("w", ctypes.c_float)
    ]

class D3DColorValue(ctypes.Structure):
    _fields_ = [
        ("r", ctypes.c_float),
        ("g", ctypes.c_float),
        ("b", ctypes.c_float),
        ("a", ctypes.c_float)
    ]

class Material(ctypes.Structure):
    _fields_ = [
        ("Diffuse", D3DColorValue),
        ("Ambient", D3DColorValue),
        ("Specular", D3DColorValue),
        ("Emissive", D3DColorValue),
        ("Power", ctypes.c_float),
        ("dwColorOp", ctypes.c_uint32),
        ("dwColorArg1", ctypes.c_uint32),
        ("dwColorArg2", ctypes.c_uint32),
        ("nRenderFlags", ctypes.c_bool),
        ("dwSrcBlend", ctypes.c_uint32),
        ("dwDestBlend", ctypes.c_uint32),
    ]

class DxtHeader(ctypes.Structure):
    _fields_ = [
        ("szID", ctypes.c_char * 4),
        ("nWidth", ctypes.c_int),
        ("nHeight", ctypes.c_int),
        ("Format", ctypes.c_int),
        ("bMipMap", ctypes.c_bool)
    ]

class ANIMATION_KEY_TYPE(Enum):
    KEY_VECTOR3 = 0
    KEY_QUATERNION = 1
    KEY_UNKNOWN = 0xffffffff

class e_PlugType(Enum):
    PLUGTYPE_NORMAL = 0
    PLUGTYPE_CLOAK = 1
    PLUGTYPE_MAX = 10
    PLUGTYPE_UNDEFINED = 0xffffffff

class D3DFORMAT(Enum):
    D3DFMT_DXT1 = 827611204
    D3DFMT_DXT2 = 844388420
    D3DFMT_DXT3 = 861165636
    D3DFMT_DXT4 = 877942852
    D3DFMT_DXT5 = 894720068
    D3DFMT_A4R4G4B4 = 26

def load_int(data, idx):
    int_val = int.from_bytes(data[idx:(idx+4)], "little", signed=True)
    idx += 4
    return int_val, idx

def load_short(data, idx):
    short_val = int.from_bytes(data[idx:(idx+2)], "little", signed=True)
    idx += 2
    return short_val, idx

def load_float(data, idx):
    int_val = int.from_bytes(data[idx:(idx+4)], "little")
    float_val = struct.unpack("f", int_val.to_bytes(4, "little"))[0]
    idx += 4
    return float_val, idx

def load_string(data, idx):
    nL, idx = load_int(data, idx)
    if nL > 0:
        str_val = data[idx:(idx+nL)].decode()
    else:
        str_val = None
    idx += nL
    return str_val, idx

def load_lodctrlvalue(data, idx):
    lodctrlvalue_val = LODCtrlValue.from_buffer_copy(data[idx:(idx+ctypes.sizeof(LODCtrlValue))])
    idx += ctypes.sizeof(LODCtrlValue)
    return lodctrlvalue_val, idx

def load_matrix44(data, idx):
    matrix44_val = Matrix44.from_buffer_copy(data[idx:(idx+ctypes.sizeof(Matrix44))])
    idx += ctypes.sizeof(Matrix44)
    return matrix44_val, idx

def load_vertex_skinned(data, idx):
    vertex_skinned_val = VertexSkinned.from_buffer_copy(data[idx:(idx+ctypes.sizeof(VertexSkinned))])
    idx += ctypes.sizeof(VertexSkinned)
    return vertex_skinned_val, idx

def load_uv(data, idx):
    uv_val = UVVector.from_buffer_copy(data[idx:(idx+ctypes.sizeof(UVVector))])
    uv_val.v = (1. - uv_val.v)
    idx += ctypes.sizeof(UVVector)
    return uv_val, idx

def load_vector(data, idx):
    vec_val = Vector.from_buffer_copy(data[idx:(idx+ctypes.sizeof(Vector))])
    idx += ctypes.sizeof(Vector)
    return vec_val, idx

def load_vertex(data, idx):
    vertex_val = Vertex.from_buffer_copy(data[idx:(idx+ctypes.sizeof(Vertex))])
    idx += ctypes.sizeof(Vertex)
    return vertex_val, idx

def load_vertexwithuv(data, idx):
    vertex_val = VertexWithUV.from_buffer_copy(data[idx:(idx+ctypes.sizeof(VertexWithUV))])
    vertex_val.v = (1. - vertex_val.v)
    idx += ctypes.sizeof(VertexWithUV)
    return vertex_val, idx

def load_quaternion(data, idx):
    quaternion_val = Quaternion.from_buffer_copy(data[idx:(idx+ctypes.sizeof(Quaternion))])
    idx += ctypes.sizeof(Quaternion)
    return quaternion_val, idx

def load_material(data, idx):
    material_val = Material.from_buffer_copy(data[idx:(idx+ctypes.sizeof(Material))])
    idx += ctypes.sizeof(Material)
    return material_val, idx

def load_dxtheader(data, idx):
    dxtheader_val = DxtHeader.from_buffer_copy(data[idx:(idx+ctypes.sizeof(DxtHeader))])
    idx += ctypes.sizeof(DxtHeader)
    return dxtheader_val, idx

class CN3BaseFileAccess:
    def __init__(self):
        self.m_szName: str = None
    
    def load(self, data, idx):
        self.m_szName, idx = load_string(data, idx)
        return idx

class CN3AnimKey:
    def __init__(self):
        self.m_nCount: int
        self.m_eType: ANIMATION_KEY_TYPE
        self.m_fSamplingRate: float
        self.m_pDatas: list = []
    
    def get_data(self, fFrm, in_val):
        if self.m_nCount <= 0:
            return in_val
        
        fD = 30.0 / self.m_fSamplingRate
        nIndex = int(fFrm * (self.m_fSamplingRate / 30.0))
        if (nIndex < 0) or (nIndex > self.m_nCount):
            return in_val
        
        fDelta = 0.0
        if nIndex == self.m_nCount:
            nIndex = self.m_nCount - 1
            fDelta = 0.0
        else:
            fDelta = (fFrm - nIndex * fD) / fD
        
        if 0.0 != fDelta:
            if self.m_eType == ANIMATION_KEY_TYPE.KEY_VECTOR3:
                v1 = self.m_pDatas[nIndex]
                v2 = self.m_pDatas[nIndex+1]
                return (mathutils.Vector([v1.x, v1.y, v1.z]) * (1.0 - fDelta)) + (mathutils.Vector([v2.x, v2.y, v2.z]) * fDelta)
            elif self.m_eType == ANIMATION_KEY_TYPE.KEY_QUATERNION:
                q1 = self.m_pDatas[nIndex]
                q2 = self.m_pDatas[nIndex+1]
                return mathutils.Quaternion([q1.w, q1.x, q1.y, q1.z]).slerp(mathutils.Quaternion([q2.w, q2.x, q2.y, q2.z]), fDelta)
        else:
            if self.m_eType == ANIMATION_KEY_TYPE.KEY_VECTOR3:
                v1 = self.m_pDatas[nIndex]
                return mathutils.Vector([v1.x, v1.y, v1.z])
            elif self.m_eType == ANIMATION_KEY_TYPE.KEY_QUATERNION:
                q1 = self.m_pDatas[nIndex]
                return mathutils.Quaternion([q1.w, q1.x, q1.y, q1.z])
    
    def load(self, data, idx):
        self.m_nCount, idx = load_int(data, idx)
        
        if self.m_nCount > 0:
            int_val, idx = load_int(data, idx)
            self.m_eType = ANIMATION_KEY_TYPE(int_val)
            self.m_fSamplingRate, idx = load_float(data, idx)
            
            if self.m_eType == ANIMATION_KEY_TYPE.KEY_VECTOR3:
                for _ in range(self.m_nCount):
                    vec_val, idx = load_vector(data, idx)
                    self.m_pDatas.append(vec_val)
                self.m_pDatas.append(vec_val)
            elif self.m_eType == ANIMATION_KEY_TYPE.KEY_QUATERNION:
                for _ in range(self.m_nCount):
                    quaternion_val, idx = load_quaternion(data, idx)
                    self.m_pDatas.append(quaternion_val)
                self.m_pDatas.append(quaternion_val)
        
        return idx

class CN3Transform(CN3BaseFileAccess):
    def __init__(self):
        super().__init__()
        
        self.m_vPos: Vector
        self.m_qRot: Quaternion
        self.m_vScale: Vector
    
        self.m_KeyPos: CN3AnimKey
        self.m_KeyRot: CN3AnimKey
        self.m_KeyScale: CN3AnimKey
    
    def load(self, data, idx):
        idx = super().load(data, idx)
        
        self.m_vPos, idx = load_vector(data, idx)
        self.m_qRot, idx = load_quaternion(data, idx)
        self.m_vScale, idx = load_vector(data, idx)
        
        self.m_KeyPos = CN3AnimKey()
        idx = self.m_KeyPos.load(data, idx)
        self.m_KeyRot = CN3AnimKey()
        idx = self.m_KeyRot.load(data, idx)
        self.m_KeyScale = CN3AnimKey()
        idx = self.m_KeyScale.load(data, idx)
        
        return idx

class CN3TransformCollision(CN3Transform):
    def __init__(self):
        super().__init__()
    
    def load(self, data, idx):
        idx = super().load(data, idx)
        
        nL, idx = load_int(data, idx)
        if nL > 0:
            raise NotImplementedError
        
        nL, idx = load_int(data, idx)
        if nL > 0:
            raise NotImplementedError
        
        return idx

class CN3Joint(CN3Transform):
    def __init__(self):
        super().__init__()
        
        self.m_KeyOrient: CN3AnimKey
        self.m_Children: list = []
    
    def load(self, data, idx):
        idx = super().load(data, idx)
        
        self.m_KeyOrient = CN3AnimKey()
        idx = self.m_KeyOrient.load(data, idx)
        
        nCC, idx = load_int(data, idx)
        for i in range(nCC):
            pChild = CN3Joint()
            idx = pChild.load(data, idx)
            self.m_Children.append(pChild)
        
        return idx

class CN3Texture(CN3BaseFileAccess):
    def __init__(self):
        super().__init__()
        
        self.HeaderOrg: DxtHeader
        self.m_iLOD: int = 0
        self.pBits: list = []
    
    def load(self, data, idx):
        idx = super().load(data, idx)
        
        self.HeaderOrg, idx = load_dxtheader(data, idx)
        
        iMMC = 1
        if self.HeaderOrg.bMipMap:
            iMMC = 0
            nW = self.HeaderOrg.nWidth
            nH = self.HeaderOrg.nHeight
            while (nW >= 4) and (nH >= 4):
                iMMC += 1
                nW //= 2
                nH //= 2
        
        if D3DFORMAT(self.HeaderOrg.Format) in [
            D3DFORMAT.D3DFMT_DXT1,
            D3DFORMAT.D3DFMT_DXT3
        ]:
            if D3DFORMAT(self.HeaderOrg.Format) == D3DFORMAT.D3DFMT_DXT1:
                iTexSize = (self.HeaderOrg.nWidth * self.HeaderOrg.nHeight // 2)
            elif D3DFORMAT(self.HeaderOrg.Format) == D3DFORMAT.D3DFMT_DXT3:
                iTexSize = (self.HeaderOrg.nWidth * self.HeaderOrg.nHeight)
            
            if iMMC > 1:
                for _ in range(iMMC):
                    self.pBits.append(data[idx:(idx + iTexSize)])
                    idx += iTexSize
                
                iWTmp, iHTmp = self.HeaderOrg.nWidth // 2, self.HeaderOrg.nHeight // 2
                while (iWTmp >= 4) and (iHTmp >= 4):
                    idx += (iWTmp * iHTmp * 2)
                    iWTmp //= 2
                    iHTmp //= 2
            else:                
                self.pBits.append(data[idx:(idx + iTexSize)])
                idx += iTexSize
                
                idx += (self.HeaderOrg.nWidth * self.HeaderOrg.nHeight // 4)
                if self.HeaderOrg.nWidth >= 1024:
                    idx += (256 * 256 * 2)
        else:
            raise NotImplementedError
        
        return idx

class CN3IMesh(CN3BaseFileAccess):
    def __init__(self):
        super().__init__()
        
        self.m_nFC: int
        self.m_nVC: int
        self.m_nUVC: int
    
        self.m_pVertices: list = []
        self.m_pwVtxIndices: list = []
    
        self.m_pfUVs: list = []
        self.m_pwUVsIndices: list = []
    
    def load(self, data, idx):
        idx = super().load(data, idx)
        
        self.m_nFC, idx = load_int(data, idx)
        self.m_nVC, idx = load_int(data, idx)
        self.m_nUVC, idx = load_int(data, idx)
        
        if self.m_nFC > 0 and self.m_nVC > 0:
            for _ in range(self.m_nVC):
                vertex_val, idx = load_vertex(data, idx)
                self.m_pVertices.append(vertex_val)
            for _ in range(3 * self.m_nFC):
                short_val, idx = load_short(data, idx)
                self.m_pwVtxIndices.append(short_val)
    
        if self.m_nUVC > 0:
            for _ in range(self.m_nUVC):
                uv_val, idx = load_uv(data, idx)
                self.m_pfUVs.append(uv_val)
            for _ in range(3 * self.m_nFC):
                short_val, idx = load_short(data, idx)
                self.m_pwUVsIndices.append(short_val)
        
        return idx

class CN3Skin(CN3IMesh):
    def __init__(self):
        super().__init__()
        
        self.m_pSkinVertices: list = []
    
    def load(self, data, idx):
        idx = super().load(data, idx)
        
        for i in range(self.m_nVC):
            vertex_skinned_val, idx = load_vertex_skinned(data, idx)
            
            pnJoints = []
            pfWeights = []
            
            if vertex_skinned_val.nAffect > 1:
                for _ in range(vertex_skinned_val.nAffect):
                    int_val, idx = load_int(data, idx)
                    pnJoints.append(int_val)
                for _ in range(vertex_skinned_val.nAffect):
                    float_val, idx = load_float(data, idx)
                    pfWeights.append(float_val)
            elif vertex_skinned_val.nAffect == 1:
                int_val, idx = load_int(data, idx)
                pnJoints.append(int_val)
            
            self.m_pSkinVertices.append({
                "vOrigin": vertex_skinned_val.vOrigin,
                "nAffect": vertex_skinned_val.nAffect,
                "pnJoints": pnJoints,
                "pfWeights": pfWeights
            })
        
        return idx

class CN3CPartSkins(CN3BaseFileAccess):
    def __init__(self):
        super().__init__()
        
        self.m_Skins: Annotated[list[CN3Skin], MAX_CHR_LOD] = [None, None, None, None]
    
    def load(self, data, idx):
        idx = super().load(data, idx)
        
        for i in range(MAX_CHR_LOD):
            skin = CN3Skin()
            idx = skin.load(data, idx)
            if skin.m_nVC > 0:
                self.m_Skins[i] = skin
            else:
                self.m_Skins[i] = None
        
        return idx

class CN3CPart(CN3BaseFileAccess):    
    def __init__(self, filepath):
        super().__init__()
        self.filepath = Path(filepath)
        
        self.m_dwReserved: int
        self.m_MtlOrg: Material
        self.m_pTexRef: CN3Texture
        self.m_pSkinsRef: CN3CPartSkins
    
    def load(self, data, idx):
        idx = super().load(data, idx)
        
        self.m_dwReserved, idx = load_int(data, idx)
        self.m_MtlOrg, idx = load_material(data, idx)
        
        szFN, idx = load_string(data, idx)
        if szFN:
            szFN = str(self.filepath.parent.joinpath(Path(szFN).name))  # Assuming the same directory
            
            f = open(szFN, 'rb')
            texture_data = f.read()
            f.close()
            
            self.m_pTexRef = CN3Texture()
            self.m_pTexRef.load(texture_data, 0)
        
        szFN, idx = load_string(data, idx)
        if szFN:
            szFN = str(self.filepath.parent.joinpath(Path(szFN).name))  # Assuming the same directory
            
            f = open(szFN, 'rb')
            skin_data = f.read()
            f.close()
            
            self.m_pSkinsRef = CN3CPartSkins()
            self.m_pSkinsRef.load(skin_data, 0)
        
        return idx

class CN3PMesh(CN3BaseFileAccess):
    def __init__(self):
        super().__init__()
        
        self.m_iNumCollapses: int
        self.m_iTotalIndexChanges: int
        self.m_iMaxNumVertices: int
        self.m_iMaxNumIndices: int
        self.m_iMinNumVertices: int
        self.m_iMinNumIndices: int
        self.m_iLODCtrlValueCount: int
        self.m_pVertices: list = []
        self.m_pIndices: list = []
        self.m_pLODCtrlValues: list = []
    
    def load(self, data, idx):
        idx = super().load(data, idx)
        
        self.m_iNumCollapses, idx = load_int(data, idx)
        self.m_iTotalIndexChanges, idx = load_int(data, idx)
        self.m_iMaxNumVertices, idx = load_int(data, idx)
        self.m_iMaxNumIndices, idx = load_int(data, idx)
        self.m_iMinNumVertices, idx = load_int(data, idx)
        self.m_iMinNumIndices, idx = load_int(data, idx)
        
        if self.m_iMaxNumVertices > 0:
            for _ in range(self.m_iMaxNumVertices):
                vector, idx = load_vertexwithuv(data, idx)
                self.m_pVertices.append(vector)
        
        if self.m_iMaxNumIndices > 0:
            for _ in range(self.m_iMaxNumIndices):
                short_val, idx = load_short(data, idx)
                self.m_pIndices.append(short_val)
        
        if self.m_iNumCollapses > 0:
            raise NotImplementedError
        
        if self.m_iTotalIndexChanges > 0:
            raise NotImplementedError
        
        self.m_iLODCtrlValueCount, idx = load_int(data, idx)
        
        if self.m_iLODCtrlValueCount > 0:
            for _ in range(self.m_iLODCtrlValueCount):
                lodctrlvalue_val, idx = load_lodctrlvalue(data, idx)
                self.m_pLODCtrlValues.append(lodctrlvalue_val)
        
        return idx

class CN3CPlugBase(CN3BaseFileAccess):
    def __init__(self):
        super().__init__()
        
        self.m_ePlugType: e_PlugType
        self.m_nJointIndex: int
        self.m_vPosition: Vector
        self.m_MtxRot: Matrix44
        self.m_vScale: Vector
        self.m_Mtl: Material
        self.m_pTexRef: CN3Texture
        self.m_PMeshInst: CN3PMesh
    
    def load(self, data, idx):
        idx = super().load(data, idx)
        
        int_val, idx = load_int(data, idx)
        self.m_ePlugType = e_PlugType(int_val)
        if self.m_ePlugType.value > e_PlugType.PLUGTYPE_MAX.value:
            self.m_ePlugType = e_PlugType.PLUGTYPE_NORMAL
        
        self.m_nJointIndex, idx = load_int(data, idx)
        self.m_vPosition, idx = load_vector(data, idx)
        self.m_MtxRot, idx = load_matrix44(data, idx)
        self.m_vScale, idx = load_vector(data, idx)
        self.m_Mtl, idx = load_material(data, idx)
        
        szFN, idx = load_string(data, idx)
        if szFN:
            szFN = str(self.filepath.parent.joinpath(Path(szFN).name))  # Assuming the same directory
            
            f = open(szFN, 'rb')
            pmesh_data = f.read()
            f.close()
            
            self.m_PMeshInst = CN3PMesh()
            self.m_PMeshInst.load(pmesh_data, 0)
        
        szFN, idx = load_string(data, idx)
        if szFN:
            szFN = str(self.filepath.parent.joinpath(Path(szFN).name))  # Assuming the same directory
            
            f = open(szFN, 'rb')
            texture_data = f.read()
            f.close()
            
            self.m_pTexRef = CN3Texture()
            self.m_pTexRef.load(texture_data, 0)
        
        return idx

class CN3CPlug(CN3CPlugBase):
    def __init__(self, filepath):
        super().__init__()
        self.filepath = Path(filepath)
        
        self.m_nTraceStep: int
    
    def load(self, data, idx):
        idx = super().load(data, idx)
        
        self.m_nTraceStep, idx = load_int(data, idx)
        
        if self.m_nTraceStep > 0:
            raise NotImplementedError
        
        iUseVMesh, idx = load_int(data, idx)
        
        if iUseVMesh:
            raise NotImplementedError
        
        return idx

class AnimData:
    def __init__(self):
        self.szName = None
        
        self.fFrmStart: float = 0.0
        self.fFrmEnd: float = 0.0
        self.fFrmPerSec: float = 30.0
        
        self.fFrmPlugTraceStart: float = 0.0
        self.fFrmPlugTraceEnd: float = 0.0
        
        self.fFrmSound0: float = 0.0
        self.fFrmSound1: float = 0.0
        
        self.fTimeBlend: float = 0.25
        self.iBlendFlags: int = 0
        
        self.fFrmStrike0: float = 0.0
        self.fFrmStrike1: float = 0.0
    
    def load(self, data, idx):
        
        nL, idx = load_int(data, idx)
        
        self.fFrmStart, idx = load_float(data, idx)
        self.fFrmEnd, idx = load_float(data, idx)
        self.fFrmPerSec, idx = load_float(data, idx)
        
        self.fFrmPlugTraceStart, idx = load_float(data, idx)
        self.fFrmPlugTraceEnd, idx = load_float(data, idx)
        
        self.fFrmSound0, idx = load_float(data, idx)
        self.fFrmSound1, idx = load_float(data, idx)
        
        self.fTimeBlend, idx = load_float(data, idx)
        self.iBlendFlags, idx = load_int(data, idx)
        
        self.fFrmStrike0, idx = load_float(data, idx)
        self.fFrmStrike1, idx = load_float(data, idx)
        
        self.szName, idx = load_string(data, idx)
        
        return idx

class CN3AnimControl:
    def __init__(self):
        super().__init__()
        
        self.m_Datas: list = []
    
    def load(self, data, idx):        
        nCount, idx = load_int(data, idx)
        
        for _ in range(nCount):
            anim_data = AnimData()
            idx = anim_data.load(data, idx)
            self.m_Datas.append(anim_data)
        
        return idx

class CN3Chr(CN3TransformCollision):    
    def __init__(self, filepath):
        super().__init__()
        self.filepath = Path(filepath)
        
        self.m_pRootJointRef: CN3Joint
        self.m_Parts: list = []
        self.m_Plugs: list = []
        
        self.m_pAniCtrlRef: CN3AnimControl
        
        self.m_nJointPartStarts: list = []
        self.m_nJointPartEnds: list = []
    
    def load(self, data, idx):
        idx = super().load(data, idx)
        
        szFN, idx = load_string(data, idx)
        szFN = str(self.filepath.parent.joinpath(Path(szFN).name))  # Assuming the same directory
        
        f = open(szFN, 'rb')
        joint_data = f.read()
        f.close()
        
        self.m_pRootJointRef = CN3Joint()
        self.m_pRootJointRef.load(joint_data, 0)
        
        iPC, idx = load_int(data, idx)
        for _ in range(iPC):
            szFN, idx = load_string(data, idx)
            szFN = str(self.filepath.parent.joinpath(Path(szFN).name))  # Assuming the same directory
            
            f = open(szFN, 'rb')
            part_data = f.read()
            f.close()
            
            part = CN3CPart(szFN)
            part.load(part_data, 0)
            self.m_Parts.append(part)
        
        iPC, idx = load_int(data, idx)
        for _ in range(iPC):
            szFN, idx = load_string(data, idx)
            szFN = str(self.filepath.parent.joinpath(Path(szFN).name))  # Assuming the same directory
            
            f = open(szFN, 'rb')
            plug_data = f.read()
            f.close()
            
            plug = CN3CPlug(szFN)
            plug.load(plug_data, 0)
            self.m_Plugs.append(plug)
        
        szFN, idx = load_string(data, idx)
        if szFN:
            szFN = str(self.filepath.parent.joinpath(Path(szFN).name))  # Assuming the same directory
            
            f = open(szFN, 'rb')
            animation_ctrl_data = f.read()
            f.close()
            
            self.m_pAniCtrlRef = CN3AnimControl()
            self.m_pAniCtrlRef.load(animation_ctrl_data, 0)
        
        for _ in range(MAX_CHR_ANI_PART):
            int_val, idx = load_int(data, idx)
            self.m_nJointPartStarts.append(int_val)
            if int_val > 0:
                raise NotImplementedError
        
        for _ in range(MAX_CHR_ANI_PART):
            int_val, idx = load_int(data, idx)
            self.m_nJointPartEnds.append(int_val)
            if int_val > 0:
                raise NotImplementedError
        
        szFN, idx = load_string(data, idx)
        if szFN:
            raise NotImplementedError
        
        return idx

# ========== END TYPES ==========

def read_some_data(context, filepath, use_some_setting):
    print("running read_some_data...")
    f = open(filepath, 'rb')
    data = f.read()
    f.close()

    # Load data    
    chr = CN3Chr(filepath)
    chr.load(data, 0)

    #
    
    iLOD = 0
    
    joints_collection = bpy.data.collections.new(f"{chr.m_szName} Joints")
    bpy.context.scene.collection.children.link(joints_collection)
    
    root_joint = chr.m_pRootJointRef
    
    armature = bpy.data.armatures.new(root_joint.m_szName)
    rig = bpy.data.objects.new(root_joint.m_szName, armature)
    joints_collection.objects.link(rig)
    
    #
    
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode='EDIT')
    
    #
    
    depth = 0
    resting_mtx = {}
    all_joints_by_idx = []
    
    def add_test_bone(resting_mtx, all_joints_by_idx, depth, joint, parent_bone, parent_mtx):
                
        m_vPos = mathutils.Vector([joint.m_vPos.x, joint.m_vPos.y, joint.m_vPos.z])
        m_qRot = mathutils.Quaternion([joint.m_qRot.w, joint.m_qRot.x, joint.m_qRot.y, joint.m_qRot.z])
        m_vScale = mathutils.Vector([joint.m_vScale.x, joint.m_vScale.y, joint.m_vScale.z])
        
        m_vPos = joint.m_KeyPos.get_data(0.0, m_vPos)
        m_qRot = joint.m_KeyRot.get_data(0.0, m_qRot)
        m_vScale = joint.m_KeyScale.get_data(0.0, m_vScale)
        
        mtx = mathutils.Matrix.LocRotScale(m_vPos, m_qRot, m_vScale)
        if parent_mtx:
            mtx = parent_mtx @ mtx
        
        depth += 1
        bone = armature.edit_bones.new(f"{depth}-{joint.m_szName}")

        bone.head = (map_mtx @ mtx).to_translation()
        bone.tail = bone.head + mathutils.Vector([0.0, -0.15, 0.0])
        
        bone.parent = parent_bone
        bone.use_connect = True
        
        resting_mtx[deepcopy(bone.name)] = [deepcopy(bone.tail), deepcopy(bone.head), deepcopy(mtx), deepcopy((bone.tail - bone.head) / 2.0 + bone.head), deepcopy(bone.matrix)]
        all_joints_by_idx.append(deepcopy(bone.name))
        
        tail_pos = mathutils.Vector([0.0, 0.0, 0.0])
        
        for child in joint.m_Children:
            depth, mtx_child = add_test_bone(resting_mtx, all_joints_by_idx, depth, child, bone, mtx)
            tail_pos += (map_mtx @ mtx_child).to_translation()
        
        if len(joint.m_Children):
            bone.tail = tail_pos / len(joint.m_Children)
        
        return depth, mtx

    add_test_bone(resting_mtx, all_joints_by_idx, depth, root_joint, None, None)
    
    #
    
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode='POSE')
    
    #
    
    depth = 0
    
    def add_test_pose(depth, frame, fFrm, joint, parent_mtx):
        
        m_vPos = mathutils.Vector([joint.m_vPos.x, joint.m_vPos.y, joint.m_vPos.z])
        m_qRot = mathutils.Quaternion([joint.m_qRot.w, joint.m_qRot.x, joint.m_qRot.y, joint.m_qRot.z])
        m_vScale = mathutils.Vector([joint.m_vScale.x, joint.m_vScale.y, joint.m_vScale.z])
    
        m_vPos = joint.m_KeyPos.get_data(fFrm, m_vPos)
        m_qRot = joint.m_KeyRot.get_data(fFrm, m_qRot)
        m_vScale = joint.m_KeyScale.get_data(fFrm, m_vScale)
    
        mtx = mathutils.Matrix.LocRotScale(m_vPos, m_qRot, m_vScale)
        if parent_mtx:
            mtx = parent_mtx @ mtx
    
        depth += 1
        bone = rig.pose.bones.get(f"{depth}-{joint.m_szName}")
    
        bone_head = (map_mtx @ mtx).to_translation()
        bone_tail = bone_head + mathutils.Vector([0.0, -0.15, 0.0])
    
        tail_pos = mathutils.Vector([0.0, 0.0, 0.0])
    
        for child in joint.m_Children:
            depth, mtx_child = add_test_pose(depth, frame, fFrm, child, mtx)
            tail_pos += (map_mtx @ mtx_child).to_translation()
    
        if len(joint.m_Children):
            bone_tail = tail_pos / len(joint.m_Children)

        rest_direction = (resting_mtx[bone.name][0] - resting_mtx[bone.name][1]).normalized()
        rotation_world = rest_direction.rotation_difference((bone_tail - bone_head).normalized())
        rot_mtx = mathutils.Matrix.LocRotScale(None, rotation_world, None)
        
        r1 = mathutils.Vector([0, 1, 0]).rotation_difference(rest_direction)
        r1 = mathutils.Matrix.LocRotScale(None, r1, None)
        r1 = rot_mtx @ r1
        
        new_rot_mtx = rig.convert_space(pose_bone=bone, matrix=r1, from_space='WORLD', to_space='LOCAL_WITH_PARENT')
        new_rot_mtx = new_rot_mtx
        
        bone.rotation_mode = 'QUATERNION'
        bone.rotation_quaternion = new_rot_mtx.to_quaternion()
        bone.keyframe_insert(data_path="rotation_quaternion", frame=frame)

        return depth, mtx

    frame, fFrm = 1, chr.m_pAniCtrlRef.m_Datas[0].fFrmStart
    while fFrm <= chr.m_pAniCtrlRef.m_Datas[0].fFrmEnd:
        add_test_pose(depth, frame, fFrm, root_joint, None)
        fFrm += 1.0; frame += 1

     #
    
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = (frame - 1)
    bpy.context.scene.frame_set(1)

    parts_collection = bpy.data.collections.new(f"{chr.m_szName} Parts")
    bpy.context.scene.collection.children.link(parts_collection)
    
    for part_idx, part in enumerate(chr.m_Parts):
        skin = part.m_pSkinsRef.m_Skins[iLOD]
        if skin is None:
            continue
        
        name = skin.m_szName
        if name is None:
            name = str(uuid4())
        
        # Mesh Verts
        mesh = bpy.data.meshes.new(f"{name}-mesh")
        
        verts = []
        for vert_idx, vert in enumerate(skin.m_pSkinVertices):
            v_origin = vert["vOrigin"]
            vert_origin = mathutils.Vector([v_origin.x, v_origin.y, v_origin.z])
            verts.append([v_origin.x, v_origin.y, v_origin.z])

        faces = []
        for i in range(0, 3 * skin.m_nFC, 3):
            faces.append([
                skin.m_pwVtxIndices[i + 0],
                skin.m_pwVtxIndices[i + 1],
                skin.m_pwVtxIndices[i + 2]
            ])
        
        mesh.from_pydata(verts, [], faces)
        mesh.update(calc_edges=True)
        
        # Object
        obj = bpy.data.objects.new(name, mesh)
        obj.data.transform(map_mtx)
        obj.data.update()
        
        # Add vertex groups for animation
        for vert_idx, vert in enumerate(skin.m_pSkinVertices):
            if vert["nAffect"] == 1:
                bone_name = all_joints_by_idx[vert["pnJoints"][0]]
                if bone_name not in obj.vertex_groups:
                    obj.vertex_groups.new(name=bone_name)
                obj.vertex_groups[bone_name].add([vert_idx], 1.0, 'REPLACE')
            elif vert["nAffect"] > 1:
                for idx, weight in zip(vert["pnJoints"], vert["pfWeights"]):
                    bone_name = all_joints_by_idx[idx]
                    if bone_name not in obj.vertex_groups:
                        obj.vertex_groups.new(name=bone_name)
                    obj.vertex_groups[bone_name].add([vert_idx], weight, 'ADD')
        
        armature_mod = obj.modifiers.new(name="Armature", type="ARMATURE")
        armature_mod.object = rig
        armature_mod.use_vertex_groups = True
        
        # Add to scene
        parts_collection.objects.link(obj)
        
        # Mesh Texture
        texture = part.m_pTexRef
        buffer = DXTBuffer(texture.HeaderOrg.nWidth, texture.HeaderOrg.nHeight)
        
        if D3DFORMAT(texture.HeaderOrg.Format) in [D3DFORMAT.D3DFMT_DXT1, D3DFORMAT.D3DFMT_DXT2]:
            with BytesIO() as img_stream:
                img_stream.write(texture.pBits[iLOD])
                img_stream.seek(0)
                buffer = buffer.DXT1Decompress(img_stream)
        elif D3DFORMAT(texture.HeaderOrg.Format) in [D3DFORMAT.D3DFMT_DXT3, D3DFORMAT.D3DFMT_DXT4]:
            with BytesIO() as img_stream:
                img_stream.write(texture.pBits[iLOD])
                img_stream.seek(0)
                buffer = buffer.DXT5Decompress(img_stream)
        else:
            raise NotImplementedError
        
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as fp:
            size = texture.HeaderOrg.nWidth if texture.HeaderOrg.nWidth < texture.HeaderOrg.nHeight else texture.HeaderOrg.nHeight
            py_image = load_texture(buffer, size, size)
            py_image.save(fp)
            image_data = bpy.data.images.load(fp.name)
        
        # Mesh Texture Coordinates
        uvlayer = mesh.uv_layers.new()
        for face in mesh.polygons:
            for loop_idx in range(face.loop_start, face.loop_start + face.loop_total):
                i = skin.m_pwUVsIndices[loop_idx]
                uvlayer.data[loop_idx].uv = [skin.m_pfUVs[i].u, skin.m_pfUVs[i].v]
        
        # Texture Material        
        material = bpy.data.materials.new(name="TextureMaterial")
        
        material.use_nodes = True
        nodes = material.node_tree.nodes
        nodes.clear()
        
        shader = nodes.new(type="ShaderNodeBsdfPrincipled")
        shader.location = 0, 0
        
        shader.inputs['Specular'].default_value = 0  # part.m_MtlOrg.Specular
        
        texture_node = nodes.new("ShaderNodeTexImage")
        texture_node.image = image_data
        texture_node.location = -400, 0
        
        material_output = nodes.new(type="ShaderNodeOutputMaterial")
        material_output.location = 400, 0
        
        material.node_tree.links.new(texture_node.outputs["Color"], shader.inputs["Base Color"])
        material.node_tree.links.new(shader.outputs["BSDF"], material_output.inputs["Surface"])
        
        if obj.data.materials:
            obj.data.materials[0] = material
        else:
            obj.data.materials.append(material)
    
    plugs_collection = bpy.data.collections.new(f"{chr.m_szName} Plugs")
    bpy.context.scene.collection.children.link(plugs_collection)
    
    for plug in chr.m_Plugs:
        pmesh = plug.m_PMeshInst
        
        name = plug.m_szName
        if name is None:
            name = str(uuid4())
        
        # Mesh Verts
        mesh = bpy.data.meshes.new(f"{name}-mesh")
        
        verts = []
        for vert in pmesh.m_pVertices:
            verts.append([vert.x, vert.y, vert.z])
        faces = []
        for i in range(0, len(pmesh.m_pIndices), 3):
            faces.append([
                pmesh.m_pIndices[i + 0],
                pmesh.m_pIndices[i + 1],
                pmesh.m_pIndices[i + 2]
            ])
        
        mesh.from_pydata(verts, [], faces)
        mesh.update(calc_edges=True)
        
        # Object
        obj = bpy.data.objects.new(name, mesh)
        obj.data.transform(map_mtx)
        obj.data.update()
        
        # Add to scene
        plugs_collection.objects.link(obj)
        
        # Mesh Texture
        texture = plug.m_pTexRef
        buffer = DXTBuffer(texture.HeaderOrg.nWidth, texture.HeaderOrg.nHeight)
        
        if D3DFORMAT(texture.HeaderOrg.Format) in [D3DFORMAT.D3DFMT_DXT1, D3DFORMAT.D3DFMT_DXT2]:
            with BytesIO() as img_stream:
                img_stream.write(texture.pBits[iLOD])
                img_stream.seek(0)
                buffer = buffer.DXT1Decompress(img_stream)
        elif D3DFORMAT(texture.HeaderOrg.Format) in [D3DFORMAT.D3DFMT_DXT3, D3DFORMAT.D3DFMT_DXT4]:
            with BytesIO() as img_stream:
                img_stream.write(texture.pBits[iLOD])
                img_stream.seek(0)
                buffer = buffer.DXT5Decompress(img_stream)
        else:
            raise NotImplementedError
        
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as fp:
            size = texture.HeaderOrg.nWidth if texture.HeaderOrg.nWidth < texture.HeaderOrg.nHeight else texture.HeaderOrg.nHeight
            py_image = load_texture(buffer, size, size)
            py_image.save(fp)
            image_data = bpy.data.images.load(fp.name)
        
        # Mesh Texture Coordinates
        uvlayer = mesh.uv_layers.new()
        for face in mesh.polygons:
            for loop_idx in range(face.loop_start, face.loop_start + face.loop_total):
                i = pmesh.m_pIndices[loop_idx]
                uvlayer.data[loop_idx].uv = [pmesh.m_pVertices[i].u, pmesh.m_pVertices[i].v]
        
        # Texture Material        
        material = bpy.data.materials.new(name="TextureMaterial")
        
        material.use_nodes = True
        nodes = material.node_tree.nodes
        nodes.clear()
        
        shader = nodes.new(type="ShaderNodeBsdfPrincipled")
        shader.location = 0, 0
        
        shader.inputs['Specular'].default_value = 0  # part.m_MtlOrg.Specular
        
        texture_node = nodes.new("ShaderNodeTexImage")
        texture_node.image = image_data
        texture_node.location = -400, 0
        
        material_output = nodes.new(type="ShaderNodeOutputMaterial")
        material_output.location = 400, 0
        
        material.node_tree.links.new(texture_node.outputs["Color"], shader.inputs["Base Color"])
        material.node_tree.links.new(shader.outputs["BSDF"], material_output.inputs["Surface"])
        
        if obj.data.materials:
            obj.data.materials[0] = material
        else:
            obj.data.materials.append(material)

    return {'FINISHED'}


class ImportKnightOnlineCharacter(Operator, ImportHelper):
    """Import a Knight Online Character file"""
    bl_idname = "import_scene.knightonline_chr"  # Changed from import_test.some_data
    bl_label = "Import Knight Online Character"   # More descriptive label

    filename_ext = ".n3chr"

    filter_glob: StringProperty(
        default="*.n3chr",
        options={'HIDDEN'},
        maxlen=255,
    )

    # Remove unused properties
    def execute(self, context):
        if not check_dependencies():
            self.report({'ERROR'}, "Please install either PIL or OpenCV from the addon preferences")
            return {'CANCELLED'}
        return read_some_data(context, self.filepath, True)


# Only needed if you want to add into a dynamic menu.
def menu_func_import(self, context):
    self.layout.operator(ImportKnightOnlineCharacter.bl_idname, 
                        text="Knight Online Character (.n3chr)")


# Register and add to the "file selector" menu (required to use F3 search "Text Import Operator" for quick access).
def register():
    # Add packages directory to Python path
    addon_dir = os.path.dirname(os.path.realpath(__file__))
    packages_dir = os.path.join(addon_dir, "packages")
    if packages_dir not in sys.path:
        sys.path.append(packages_dir)

    # Register classes
    bpy.utils.register_class(KnightOnlinePreferences)
    bpy.utils.register_class(InstallDependenciesOperator)
    bpy.utils.register_class(ImportKnightOnlineCharacter)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)

def unregister():
    # Unregister classes
    bpy.utils.unregister_class(KnightOnlinePreferences)
    bpy.utils.unregister_class(InstallDependenciesOperator)
    bpy.utils.unregister_class(ImportKnightOnlineCharacter)
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)

    # Remove packages directory from Python path
    addon_dir = os.path.dirname(os.path.realpath(__file__))
    packages_dir = os.path.join(addon_dir, "packages")
    if packages_dir in sys.path:
        sys.path.remove(packages_dir)

if __name__ == "__main__":
    register()
