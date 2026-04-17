def dibujar_inspeccion_2d(inputs, df_simulacion, año, px0_valor):
    # 1. Extraer datos del año seleccionado
    fila = df_simulacion[df_simulacion["Tiempo (y)"] == año].iloc[0]
    px_actual = fila["Px (mm)"]
    b_actual = fila["b"]
    d_actual = fila["d"]
    phi_actual = fila["phi1 (mm)"]
    
    # Inputs originales
    b_0 = inputs['ancho_b']
    d_0 = inputs['canto_d']
    recu = inputs['recubrimiento']
    n_b = int(inputs['n_barras'])
    phi_0 = inputs['phi_base']

    fig, ax = plt.subplots(figsize=(5, 7))
    
    # 2. Dibujar contorno original (referencia fantasma)
    rect_fantom = plt.Rectangle((0, 0), b_0, d_0, linewidth=1, 
                                 edgecolor='gray', facecolor='none', ls=':')
    ax.add_patch(rect_fantom)

    # 3. Dibujar Hormigón actual
    # Si b_actual < b_0, centramos la sección (desprendimiento)
    off_x = (b_0 - b_actual) / 2
    color_hormigon = 'lightgrey'
    rect_h = plt.Rectangle((off_x, 0), b_actual, d_actual, 
                            linewidth=2, edgecolor='black', facecolor=color_hormigon)
    ax.add_patch(rect_h)

    # 4. Lógica de FISURACIÓN (Si Px > Px0 y aún no hay desprendimiento total)
    if px_actual >= px0_valor and d_actual == d_0:
        # Dibujamos unas líneas que simulan fisuras en el recubrimiento inferior
        for i in range(n_b):
            x_f = (b_0 / (n_b + 1)) * (i + 1)
            ax.plot([x_f, x_f], [0, recu], color='black', lw=1.5, alpha=0.8) # Fisura vertical
            ax.plot([x_f-5, x_f+5], [recu/2, recu/2], color='black', lw=0.8) # Ramificación

    # 5. Dibujar Armaduras Inferiores (Degradándose)
    for i in range(n_b):
        x_pos = (b_0 / (n_b + 1)) * (i + 1)
        # Color: De rojo (sano) a marrón oscuro (oxidado)
        color_acero = 'red' if px_actual == 0 else 'brown'
        
        # El radio se reduce según phi_actual
        circ = plt.Circle((x_pos, recu), phi_actual/2, 
                          facecolor=color_acero, edgecolor='black', lw=1)
        ax.add_patch(circ)

    # Ajustes de visualización
    ax.set_xlim(-b_0*0.1, b_0*1.1)
    ax.set_ylim(-d_0*0.1, d_0*1.1)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Leyenda informativa
    estado = "SANO"
    if px_actual >= px0_valor: estado = "FISURADO"
    if d_actual < d_0: estado = "DESPRENDIDO"
    
    ax.set_title(f"Año {año} - Estado: {estado}\n$P_x$ actual: {px_actual:.3f} mm", 
                 fontsize=10, fontweight='bold')
    
    return fig
