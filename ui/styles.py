"""
Enhanced CSS styling for the Health AI Chatbot application.

This module provides:
- Accessible, WCAG AA compliant styles
- Responsive design for mobile/tablet/desktop
- Performance-optimized animations
- Dual theme system (dark/light)
- Consistent design system
"""


def get_custom_css(theme: str = "dark") -> str:
    """
    Get enhanced custom CSS for the application with theme support.

    Args:
        theme: "dark" or "light" theme mode

    Returns:
        str: Optimized CSS with accessibility, responsive design, and theming
    """
    # Theme-specific color variables
    if theme == "light":
        theme_vars = """
            /* Light Theme Colors */
            --primary: #0891b2;
            --primary-light: #06b6d4;
            --primary-dark: #0e7490;
            --secondary: #7c3aed;
            --secondary-light: #a78bfa;
            --success: #059669;
            --warning: #d97706;
            --error: #dc2626;
            --info: #2563eb;

            /* Neutrals - Light */
            --bg-primary: #ffffff;
            --bg-secondary: #f8fafc;
            --bg-tertiary: #f1f5f9;
            --bg-card: #ffffff;
            --text-primary: #0f172a;
            --text-secondary: #475569;
            --text-tertiary: #64748b;
            --border-light: rgba(148, 163, 184, 0.2);
            --border-medium: rgba(148, 163, 184, 0.4);
            --shadow-color: rgba(0, 0, 0, 0.1);

            /* Gradients - Light */
            --gradient-hero: linear-gradient(135deg, #e0f2fe 0%, #ddd6fe 100%);
            --gradient-card: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
            --gradient-primary: linear-gradient(135deg, #0891b2, #7c3aed);
        """
    else:
        theme_vars = """
            /* Dark Theme Colors */
            --primary: #00d9ff;
            --primary-light: #4ee4ff;
            --primary-dark: #00b8d4;
            --secondary: #7c3aed;
            --secondary-light: #a78bfa;
            --success: #10b981;
            --warning: #fbbf24;
            --error: #ef4444;
            --info: #3b82f6;

            /* Neutrals - Dark */
            --bg-primary: rgba(10, 14, 26, 0.95);
            --bg-secondary: rgba(21, 29, 53, 0.85);
            --bg-tertiary: rgba(26, 35, 66, 0.7);
            --bg-card: rgba(30, 41, 59, 0.5);
            --text-primary: #f1f5f9;
            --text-secondary: #cbd5e1;
            --text-tertiary: #94a3b8;
            --border-light: rgba(100, 116, 255, 0.2);
            --border-medium: rgba(100, 116, 255, 0.3);
            --shadow-color: rgba(0, 0, 0, 0.4);

            /* Gradients - Dark */
            --gradient-hero: linear-gradient(135deg, rgba(26, 35, 66, 0.7), rgba(21, 29, 53, 0.85));
            --gradient-card: linear-gradient(135deg, rgba(26, 35, 66, 0.5), rgba(21, 29, 53, 0.5));
            --gradient-primary: linear-gradient(135deg, #00d9ff, #7c3aed);
        """

    # Use regular string concatenation instead of f-string to avoid escaping all braces
    return """<style>
        /* ============================================
           DESIGN TOKENS & CSS VARIABLES
        ============================================ */
        :root {
            """ + theme_vars + """

            /* Spacing Scale (8pt grid) */
            --space-xs: 0.5rem;
            --space-sm: 0.75rem;
            --space-md: 1rem;
            --space-lg: 1.5rem;
            --space-xl: 2rem;
            --space-2xl: 3rem;

            /* Typography Scale */
            --font-xs: 0.75rem;
            --font-sm: 0.875rem;
            --font-base: 1rem;
            --font-lg: 1.125rem;
            --font-xl: 1.25rem;
            --font-2xl: 1.5rem;
            --font-3xl: 2rem;
            --font-4xl: 2.5rem;

            /* Border Radius */
            --radius-sm: 8px;
            --radius-md: 12px;
            --radius-lg: 16px;
            --radius-xl: 24px;
            --radius-full: 9999px;

            /* Shadows */
            --shadow-sm: 0 2px 4px rgba(0, 0, 0, 0.1);
            --shadow-md: 0 4px 12px rgba(0, 0, 0, 0.15);
            --shadow-lg: 0 8px 24px rgba(0, 0, 0, 0.2);
            --shadow-xl: 0 20px 60px rgba(0, 0, 0, 0.4);

            /* Transitions */
            --transition-fast: 150ms cubic-bezier(0.4, 0, 0.2, 1);
            --transition-base: 250ms cubic-bezier(0.4, 0, 0.2, 1);
            --transition-slow: 350ms cubic-bezier(0.4, 0, 0.2, 1);
        }

        /* ============================================
           ACCESSIBILITY IMPROVEMENTS
        ============================================ */

        /* Focus visible for keyboard navigation */
        *:focus-visible {
            outline: 3px solid var(--primary);
            outline-offset: 2px;
            border-radius: var(--radius-sm);
        }

        /* Skip to main content link */
        .skip-to-main {
            position: absolute;
            top: -40px;
            left: 0;
            background: var(--primary);
            color: var(--bg-primary);
            padding: var(--space-sm) var(--space-md);
            text-decoration: none;
            border-radius: var(--radius-sm);
            z-index: 100;
            font-weight: 600;
        }

        .skip-to-main:focus {
            top: var(--space-sm);
        }

        /* Improved text contrast */
        .high-contrast {
            color: var(--text-primary);
            font-weight: 500;
        }

        /* Minimum touch target size (44x44px for accessibility) */
        button, a, .clickable {
            min-height: 44px;
            min-width: 44px;
        }

        /* ============================================
           THEME TOGGLE BUTTON
        ============================================ */
        .theme-toggle {
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 1000;
            background: var(--bg-card);
            border: 1.5px solid var(--border-medium);
            border-radius: var(--radius-full);
            padding: var(--space-sm) var(--space-md);
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: var(--space-xs);
            font-weight: 600;
            color: var(--text-primary);
            transition: all var(--transition-base);
            box-shadow: var(--shadow-md);
        }

        .theme-toggle:hover {
            transform: translateY(-2px);
            box-shadow: var(--shadow-lg);
            border-color: var(--primary);
        }

        /* ============================================
           HERO SECTION - RESPONSIVE
        ============================================ */
        .app-hero {
            display: grid;
            grid-template-columns: 1fr;
            gap: var(--space-xl);
            padding: var(--space-2xl);
            border-radius: var(--radius-xl);
            border: 1.5px solid var(--border-light);
            background: var(--gradient-hero);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            box-shadow: 0 20px 60px var(--shadow-color),
                        0 8px 16px rgba(124, 58, 237, 0.1),
                        inset 0 1px 0 rgba(255, 255, 255, 0.05);
            margin-bottom: var(--space-xl);
            position: relative;
            overflow: hidden;
        }

        /* Desktop: side-by-side layout */
        @media (min-width: 768px) {
            .app-hero {
                grid-template-columns: 2fr 1fr;
            }
        }

        /* Tablet optimization */
        @media (max-width: 767px) {
            .app-hero {
                padding: var(--space-lg);
                gap: var(--space-lg);
            }
        }

        /* Mobile optimization */
        @media (max-width: 480px) {
            .app-hero {
                padding: var(--space-md);
                gap: var(--space-md);
                border-radius: var(--radius-lg);
            }
        }

        /* Optimized glow animation - reduced for performance */
        .app-hero::before {
            content: '';
            position: absolute;
            top: -50%;
            right: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(
                circle at center,
                rgba(0, 217, 255, 0.06) 0%,
                rgba(124, 58, 237, 0.04) 50%,
                transparent 70%
            );
            animation: heroGlow 15s ease-in-out infinite;
            will-change: transform;
        }

        /* Reduced motion for users who prefer it */
        @media (prefers-reduced-motion: reduce) {
            .app-hero::before {
                animation: none;
            }
        }

        @keyframes heroGlow {
            0%, 100% { transform: translate(0, 0) rotate(0deg); }
            50% { transform: translate(-5%, 5%) rotate(90deg); }
        }

        .hero-left {
            position: relative;
            z-index: 1;
        }

        .hero-left h1 {
            margin-bottom: var(--space-sm);
            font-size: var(--font-3xl);
            font-weight: 800;
            background: var(--gradient-primary);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            line-height: 1.2;
        }

        @media (min-width: 768px) {
            .hero-left h1 {
                font-size: var(--font-4xl);
            }
        }

        .hero-pill {
            display: inline-flex;
            align-items: center;
            gap: var(--space-xs);
            padding: var(--space-xs) var(--space-md);
            border-radius: var(--radius-full);
            background: linear-gradient(135deg, rgba(0, 217, 255, 0.15), rgba(124, 58, 237, 0.15));
            border: 1px solid var(--border-medium);
            color: var(--primary);
            font-weight: 700;
            font-size: var(--font-xs);
            letter-spacing: 0.08em;
            text-transform: uppercase;
            box-shadow: var(--shadow-sm);
        }

        .hero-subtitle {
            font-size: var(--font-base);
            color: var(--text-secondary);
            margin-top: var(--space-md);
            margin-bottom: var(--space-lg);
            line-height: 1.7;
            font-weight: 400;
        }

        @media (min-width: 768px) {
            .hero-subtitle {
                font-size: var(--font-lg);
            }
        }

        .hero-steps {
            display: flex;
            flex-wrap: wrap;
            gap: var(--space-sm);
        }

        .hero-step {
            background: linear-gradient(135deg, rgba(0, 217, 255, 0.12), rgba(124, 58, 237, 0.08));
            border: 1px solid var(--border-light);
            color: var(--primary);
            font-weight: 600;
            border-radius: var(--radius-md);
            padding: var(--space-sm) var(--space-md);
            font-size: var(--font-sm);
            transition: all var(--transition-base);
            backdrop-filter: blur(10px);
        }

        .hero-step:hover {
            transform: translateY(-2px);
            box-shadow: var(--shadow-md);
            border-color: var(--primary);
        }

        .hero-highlight {
            background: var(--gradient-card);
            border-radius: var(--radius-xl);
            border: 1.5px solid var(--border-medium);
            padding: var(--space-xl);
            display: flex;
            flex-direction: column;
            gap: var(--space-md);
            box-shadow: var(--shadow-lg),
                        inset 0 1px 0 rgba(255, 255, 255, 0.05);
            position: relative;
            z-index: 1;
        }

        @media (max-width: 767px) {
            .hero-highlight {
                padding: var(--space-lg);
            }
        }

        .hero-metric-label {
            text-transform: uppercase;
            font-size: var(--font-xs);
            letter-spacing: 0.15em;
            color: var(--text-tertiary);
            font-weight: 600;
        }

        .hero-metric-value {
            font-size: 3rem;
            font-weight: 800;
            color: var(--text-primary);
            line-height: 1;
            text-shadow: 0 2px 12px rgba(0, 217, 255, 0.3);
        }

        @media (min-width: 768px) {
            .hero-metric-value {
                font-size: 3.5rem;
            }
        }

        .hero-metric-badge {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: var(--space-sm) var(--space-md);
            border-radius: var(--radius-full);
            font-weight: 700;
            font-size: var(--font-sm);
            letter-spacing: 0.02em;
            box-shadow: var(--shadow-md);
        }

        .hero-highlight-note {
            font-size: var(--font-sm);
            color: var(--text-secondary);
            line-height: 1.6;
        }

        /* ============================================
           RISK BADGES - IMPROVED CONTRAST
        ============================================ */
        .risk-badge-low {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.3), rgba(5, 150, 105, 0.25));
            border: 1px solid rgba(16, 185, 129, 0.5);
            color: #d1fae5;
            box-shadow: var(--shadow-md);
        }

        .risk-badge-medium {
            background: linear-gradient(135deg, rgba(251, 191, 36, 0.3), rgba(245, 158, 11, 0.25));
            border: 1px solid rgba(251, 191, 36, 0.5);
            color: #fef3c7;
            box-shadow: var(--shadow-md);
        }

        .risk-badge-high {
            background: linear-gradient(135deg, rgba(248, 113, 113, 0.3), rgba(239, 68, 68, 0.25));
            border: 1px solid rgba(248, 113, 113, 0.5);
            color: #fee2e2;
            box-shadow: var(--shadow-md);
        }

        .risk-badge-neutral {
            background: linear-gradient(135deg, rgba(0, 217, 255, 0.3), rgba(124, 58, 237, 0.25));
            border: 1px solid rgba(0, 217, 255, 0.5);
            color: #e0f2fe;
            box-shadow: var(--shadow-md);
        }

        /* ============================================
           METRIC CARDS - ENHANCED
        ============================================ */
        .metric-card {
            background: var(--gradient-card);
            border: 1px solid var(--border-light);
            border-radius: var(--radius-lg);
            padding: var(--space-lg);
            transition: all var(--transition-base);
        }

        .metric-card:hover {
            border-color: var(--border-medium);
            box-shadow: 0 8px 24px var(--shadow-color);
            transform: translateY(-2px);
        }

        /* ============================================
           AGENT SELECTOR CARD
        ============================================ */
        .agent-card {
            background: var(--bg-card);
            border: 2px solid var(--border-light);
            border-radius: var(--radius-lg);
            padding: var(--space-md);
            margin: var(--space-sm) 0;
            cursor: pointer;
            transition: all var(--transition-fast);
        }

        .agent-card:hover {
            border-color: var(--primary);
            box-shadow: 0 4px 12px var(--shadow-color);
            transform: translateX(4px);
        }

        .agent-card.active {
            border-color: var(--primary);
            background: linear-gradient(135deg, rgba(0, 217, 255, 0.1), rgba(124, 58, 237, 0.05));
            box-shadow: 0 0 20px rgba(0, 217, 255, 0.2);
        }

        .agent-icon {
            font-size: 2rem;
            margin-bottom: var(--space-xs);
        }

        .agent-name {
            font-weight: 700;
            color: var(--text-primary);
            margin-bottom: var(--space-xs);
        }

        .agent-description {
            font-size: var(--font-sm);
            color: var(--text-secondary);
            line-height: 1.5;
        }

        .agent-status {
            display: inline-flex;
            align-items: center;
            gap: var(--space-xs);
            padding: 0.25rem 0.75rem;
            border-radius: var(--radius-full);
            font-size: var(--font-xs);
            font-weight: 600;
            margin-top: var(--space-sm);
        }

        .agent-status.ready {
            background: rgba(16, 185, 129, 0.2);
            color: var(--success);
            border: 1px solid var(--success);
        }

        .agent-status.loading {
            background: rgba(251, 191, 36, 0.2);
            color: var(--warning);
            border: 1px solid var(--warning);
        }

        /* ============================================
           INFO BOXES & ALERTS
        ============================================ */
        .info-box {
            background: linear-gradient(135deg, rgba(0, 217, 255, 0.1), rgba(124, 58, 237, 0.1));
            border-left: 4px solid var(--primary);
            padding: var(--space-md);
            margin: var(--space-md) 0;
            border-radius: var(--radius-sm);
        }

        .info-box p {
            color: var(--text-secondary);
            margin: 0;
            font-size: var(--font-sm);
            line-height: 1.6;
        }

        /* ============================================
           ACTION CARDS - INSIGHT STYLING
        ============================================ */
        .action-card {
            background: var(--bg-card);
            border-left: 3px solid var(--secondary);
            padding: var(--space-md);
            margin: var(--space-sm) 0;
            border-radius: var(--radius-sm);
            transition: all var(--transition-base);
            box-shadow: 0 2px 8px var(--shadow-color);
        }

        .action-card:hover {
            background: var(--bg-tertiary);
            border-left-width: 4px;
            transform: translateX(4px);
            box-shadow: 0 4px 12px var(--shadow-color);
        }

        .action-card-icon {
            color: var(--primary);
            font-size: var(--font-xl);
            margin-bottom: var(--space-xs);
        }

        .action-card-title {
            color: var(--text-primary);
            font-weight: 600;
            margin-bottom: var(--space-xs);
        }

        .action-card-impact {
            color: var(--warning);
            font-size: var(--font-sm);
            margin: var(--space-xs) 0;
            padding-left: calc(var(--font-xl) + var(--space-xs));
        }

        .action-card-action {
            color: var(--success);
            font-size: var(--font-sm);
            margin-top: var(--space-xs);
            padding-left: calc(var(--font-xl) + var(--space-xs));
            font-weight: 600;
        }

        /* ============================================
           BUTTONS - ENHANCED
        ============================================ */
        .stButton > button {
            border-radius: var(--radius-md);
            font-weight: 600;
            transition: all var(--transition-base);
            box-shadow: var(--shadow-sm);
        }

        .stButton > button:hover {
            box-shadow: var(--shadow-md);
            transform: translateY(-1px);
        }

        .stButton > button:active {
            transform: translateY(0);
        }

        /* ============================================
           RESPONSIVE UTILITIES
        ============================================ */

        /* Hide on mobile */
        @media (max-width: 767px) {
            .hide-mobile {
                display: none !important;
            }
        }

        /* Show only on mobile */
        @media (min-width: 768px) {
            .show-mobile {
                display: none !important;
            }
        }

        /* ============================================
           LOADING STATES
        ============================================ */
        .loading-skeleton {
            background: linear-gradient(
                90deg,
                var(--bg-secondary) 0%,
                var(--bg-tertiary) 50%,
                var(--bg-secondary) 100%
            );
            background-size: 200% 100%;
            animation: loading 1.5s ease-in-out infinite;
            border-radius: var(--radius-sm);
        }

        @keyframes loading {
            0% { background-position: 200% 0; }
            100% { background-position: -200% 0; }
        }

        /* ============================================
           PRINT STYLES
        ============================================ */
        @media print {
            .app-hero::before,
            .hero-glow,
            .no-print {
                display: none !important;
            }

            body {
                background: white !important;
                color: black !important;
            }
        }

        /* ============================================
           PROGRESS INDICATOR
        ============================================ */
        .progress-indicator {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin: 2rem 0;
            padding: 1rem;
            background: rgba(26, 35, 66, 0.5);
            border-radius: 12px;
        }

        .progress-step {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 0.5rem;
            flex: 1;
            position: relative;
        }

        .progress-step:not(:last-child)::after {
            content: '';
            position: absolute;
            top: 20px;
            left: 60%;
            width: 80%;
            height: 2px;
            background: rgba(100, 116, 255, 0.2);
        }

        .progress-step-complete::after {
            background: #10b981 !important;
        }

        .progress-step-circle {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            transition: all 0.3s ease;
        }

        .progress-step-pending .progress-step-circle {
            background: rgba(100, 116, 255, 0.1);
            border: 2px solid rgba(100, 116, 255, 0.3);
            color: #94a3b8;
        }

        .progress-step-active .progress-step-circle {
            background: linear-gradient(135deg, #00d9ff, #7c3aed);
            border: 2px solid #00d9ff;
            color: white;
            box-shadow: 0 4px 12px rgba(0, 217, 255, 0.3);
        }

        .progress-step-complete .progress-step-circle {
            background: #10b981;
            border: 2px solid #059669;
            color: white;
        }

        .progress-step-label {
            font-size: 0.875rem;
            font-weight: 600;
            text-align: center;
        }

        .progress-step-pending .progress-step-label {
            color: #94a3b8;
        }

        .progress-step-active .progress-step-label {
            color: #00d9ff;
        }

        .progress-step-complete .progress-step-label {
            color: #10b981;
        }

        @media (max-width: 640px) {
            .progress-step-label {
                font-size: 0.75rem;
            }

            .progress-step-circle {
                width: 32px;
                height: 32px;
                font-size: 0.875rem;
            }}
        }}

        /* ============================================
           STREAMLIT OVERRIDES
        ============================================ */
        /* Main content background */
        .main {{
            background-color: var(--bg-primary);
        }}

        /* Sidebar styling */
        section[data-testid="stSidebar"] {{
            background-color: var(--bg-secondary);
            border-right: 1px solid var(--border-light);
        }}

        /* Chat container */
        .stChatFloatingInputContainer {{
            background-color: var(--bg-secondary);
            border-top: 1px solid var(--border-light);
        }}

        /* Input fields */
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input,
        .stSelectbox > div > div {{
            background-color: var(--bg-card);
            color: var(--text-primary);
            border-color: var(--border-light);
        }}

        .stTextInput > div > div > input:focus,
        .stNumberInput > div > div > input:focus,
        .stSelectbox > div > div:focus-within {{
            border-color: var(--primary);
            box-shadow: 0 0 0 2px rgba(0, 217, 255, 0.2);
        }}
    </style>"""


def get_theme_toggle_html(current_theme: str = "dark") -> str:
    """
    Generate HTML for theme toggle button.

    Args:
        current_theme: Current theme ("dark" or "light")

    Returns:
        str: HTML for theme toggle button
    """
    icon = "☀️" if current_theme == "dark" else "🌙"
    text = "Light Mode" if current_theme == "dark" else "Dark Mode"

    return f"""<div class="theme-toggle" onclick="window.parent.postMessage({{type: 'streamlit:setComponentValue', value: '{{'clicked': true}}'}}', '*')">
        <span style="font-size: 1.2rem;">{icon}</span>
        <span>{text}</span>
    </div>"""


def get_progress_indicator_html(current_step: int, total_steps: int = 3) -> str:
    """
    Generate HTML for a progress indicator.

    Args:
        current_step: Current step number (1-indexed)
        total_steps: Total number of steps

    Returns:
        str: HTML for progress indicator (styles are in get_custom_css())
    """
    steps = []
    step_names = ["Assessment", "Analysis", "AI Guidance"]

    for i in range(1, total_steps + 1):
        status = "complete" if i < current_step else "active" if i == current_step else "pending"
        step_name = step_names[i-1] if i <= len(step_names) else f"Step {i}"

        steps.append(f"""<div class="progress-step progress-step-{status}">
            <div class="progress-step-circle">{i}</div>
            <div class="progress-step-label">{step_name}</div>
        </div>""")

    return f"""<div class="progress-indicator">
        {''.join(steps)}
    </div>"""


__all__ = ['get_custom_css', 'get_progress_indicator_html', 'get_theme_toggle_html']
