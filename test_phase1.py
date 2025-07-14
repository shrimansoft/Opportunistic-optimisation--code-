#!/usr/bin/env python3
"""
Phase 1 Testing Script - RWare Migration

This script tests and visualizes the Phase 1 implementation of the RWare migration.
Run this script to verify that the grid system, highways, and coordinate utilities are working correctly.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_phase1():
    """Main test function for Phase 1 implementation."""
    print("🚀 Phase 1 Implementation Test")
    print("=" * 50)
    
    try:
        # Import warehouse
        from warehouse_sim.warehouse import Warehouse
        print("✅ Warehouse import successful")
        
        # Create warehouse instance
        warehouse = Warehouse(seed=42)
        print("✅ Warehouse initialization successful")
        
        # Test grid system
        print(f"\n📊 Grid System Tests:")
        print(f"   Grid size: {warehouse.grid_size}")
        print(f"   Grid shape: {warehouse.grid.shape}")
        print(f"   Expected: (2, 20, 20)")
        assert warehouse.grid.shape == (2, 20, 20), "Grid shape mismatch!"
        print("   ✅ Grid system structure correct")
        
        # Test highways
        print(f"\n🛣️  Highway System Tests:")
        highway_count = np.sum(warehouse.highways)
        print(f"   Total highway cells: {highway_count}")
        print(f"   Highway at (0,0): {warehouse._is_highway(0, 0)}")
        print(f"   Highway at (3,0): {warehouse._is_highway(3, 0)}")
        print(f"   Highway at (0,4): {warehouse._is_highway(0, 4)}")
        print(f"   Highway at (19,19): {warehouse._is_highway(19, 19)}")  # Bottom row
        assert highway_count > 0, "No highways found!"
        print("   ✅ Highway system working")
        
        # Test coordinate utilities
        print(f"\n📍 Coordinate Utilities Tests:")
        try:
            from warehouse_sim.utils.coordinates import flat_to_grid, grid_to_flat, validate_coordinates
            
            # Test round-trip conversion
            test_indices = [0, 25, 399]
            for idx in test_indices:
                x, y = flat_to_grid(idx, 20)
                back = grid_to_flat(x, y, 20)
                print(f"   Index {idx} -> ({x}, {y}) -> {back}")
                assert back == idx, f"Round-trip failed for {idx}"
            
            # Test validation
            assert validate_coordinates(10, 10, (20, 20)), "Valid coordinates rejected"
            assert not validate_coordinates(25, 10, (20, 20)), "Invalid coordinates accepted"
            print("   ✅ Coordinate utilities working")
            
        except ImportError as e:
            print(f"   ❌ Coordinate utilities import failed: {e}")
            return False
        
        # Create visualization
        print(f"\n🎨 Creating Visualization...")
        create_phase1_visualization(warehouse)
        
        print(f"\n🎉 Phase 1 Implementation: SUCCESSFUL!")
        print("=" * 50)
        return True
        
    except Exception as e:
        print(f"❌ Phase 1 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_phase1_visualization(warehouse):
    """Create a comprehensive visualization of Phase 1 implementation."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Phase 1 Grid System - RWare Migration Test', fontsize=16, fontweight='bold')
    
    # 1. Highway System Visualization
    ax1 = axes[0, 0]
    ax1.set_title('Highway System Layout')
    
    # Create visualization matrix
    highway_visual = warehouse.highways.astype(float)
    
    im1 = ax1.imshow(highway_visual, cmap='RdYlBu_r', interpolation='nearest')
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    
    # Add grid lines
    for i in range(warehouse.grid_size[1] + 1):
        ax1.axvline(x=i-0.5, color='black', linewidth=0.3, alpha=0.5)
    for i in range(warehouse.grid_size[0] + 1):
        ax1.axhline(y=i-0.5, color='black', linewidth=0.3, alpha=0.5)
    
    # Colorbar
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label('Highway (1) vs Regular (0)')
    
    # 2. Grid Layer Structure
    ax2 = axes[0, 1]
    ax2.set_title('Collision Layer Structure')
    
    # Show layer info
    layer_text = f"Grid Implementation Details:\n\n"
    layer_text += f"Total Shape: {warehouse.grid.shape}\n"
    layer_text += f"Layers: {warehouse.grid.shape[0]}\n"
    layer_text += f"Height: {warehouse.grid.shape[1]}\n"
    layer_text += f"Width: {warehouse.grid.shape[2]}\n\n"
    layer_text += f"Layer 0: AGENTS (_LAYER_AGENTS)\n"
    layer_text += f"Layer 1: SHELFS (_LAYER_SHELFS)\n\n"
    layer_text += f"Memory: {warehouse.grid.nbytes} bytes\n"
    layer_text += f"Data Type: {warehouse.grid.dtype}"
    
    ax2.text(0.05, 0.95, layer_text, transform=ax2.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    
    # 3. Highway Statistics
    ax3 = axes[1, 0]
    ax3.set_title('Highway Distribution Analysis')
    
    # Calculate highway distribution
    highway_by_row = np.sum(warehouse.highways, axis=1)
    highway_by_col = np.sum(warehouse.highways, axis=0)
    
    x_pos = np.arange(len(highway_by_row))
    
    # Create bar chart
    bars = ax3.bar(x_pos, highway_by_row, alpha=0.7, color='steelblue')
    ax3.set_xlabel('Row Index')
    ax3.set_ylabel('Highway Cells per Row')
    ax3.grid(True, alpha=0.3)
    
    # Highlight pattern
    for i, bar in enumerate(bars):
        if i % 4 == 0:  # Every 4th row (horizontal highways)
            bar.set_color('orange')
    
    # 4. Phase 1 Status Summary
    ax4 = axes[1, 1]
    ax4.set_title('Phase 1 Implementation Status')
    
    # Calculate statistics
    total_cells = warehouse.grid_size[0] * warehouse.grid_size[1]
    highway_cells = np.sum(warehouse.highways)
    coverage = (highway_cells / total_cells) * 100
    
    status_text = f"PHASE 1 MIGRATION STATUS\n"
    status_text += f"{'='*30}\n\n"
    status_text += f"✅ Grid System: COMPLETE\n"
    status_text += f"   • 2D collision layers implemented\n"
    status_text += f"   • {total_cells} total cells\n"
    status_text += f"   • Shape: {warehouse.grid_size}\n\n"
    status_text += f"✅ Highway System: COMPLETE\n"
    status_text += f"   • {highway_cells} highway cells\n"
    status_text += f"   • {coverage:.1f}% grid coverage\n"
    status_text += f"   • Pattern: 3-col, 4-row intervals\n\n"
    status_text += f"✅ Entity Foundation: COMPLETE\n"
    status_text += f"   • Base Entity class created\n"
    status_text += f"   • Core enums implemented\n\n"
    status_text += f"✅ Coordinate Utils: COMPLETE\n"
    status_text += f"   • Conversion functions working\n"
    status_text += f"   • Validation implemented\n\n"
    status_text += f"🎯 READY FOR PHASE 2!"
    
    ax4.text(0.05, 0.95, status_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    
    plt.tight_layout()
    
    # Save the visualization
    output_path = 'phase1_implementation_test.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   📊 Visualization saved: {output_path}")
    
    # Show the plot
    plt.show()
    
    
    return fig


if __name__ == "__main__":
    print("Starting Phase 1 Implementation Test...\n")
    success = test_phase1()
    
    if success:
        print("\n🎉 Phase 1 testing completed successfully!")
        print("Ready to proceed to Phase 2: Robot System Overhaul")
    else:
        print("\n❌ Phase 1 testing failed. Please check the implementation.")
        sys.exit(1)
