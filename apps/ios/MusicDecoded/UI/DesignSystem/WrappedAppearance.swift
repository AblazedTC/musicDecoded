import SwiftUI
import UIKit

/// UIKit chrome in the "Wrapped" style (DESIGN.md §1): the tab bar is ink with a 3pt black top
/// rule, lime selected icon + label and `textMuted` unselected; navigation bars are ink with
/// Archivo Black titles. Applied once at launch.
@MainActor
enum WrappedAppearance {
    static func apply() {
        applyTabBar()
        applyNavigationBar()
    }

    private static func applyTabBar() {
        let appearance = UITabBarAppearance()
        appearance.configureWithOpaqueBackground()
        appearance.backgroundColor = UIColor(Palette.ink)
        // A 3pt image instead of the 1px hairline; the color also tints it if UIKit treats it as a template.
        appearance.shadowColor = UIColor(Palette.border)
        appearance.shadowImage = rule(height: BorderWidth.thick, color: UIColor(Palette.border))

        let muted = UIColor(Palette.textMuted)
        let lime = UIColor(Palette.lime)
        let label = scaledFont(FontName.bold, size: 11, style: .caption2)
        let item = UITabBarItemAppearance()
        item.normal.iconColor = muted
        item.normal.titleTextAttributes = [.foregroundColor: muted, .font: label, .kern: 1.1]
        item.selected.iconColor = lime
        item.selected.titleTextAttributes = [.foregroundColor: lime, .font: label, .kern: 1.1]
        appearance.stackedLayoutAppearance = item
        appearance.inlineLayoutAppearance = item
        appearance.compactInlineLayoutAppearance = item

        UITabBar.appearance().standardAppearance = appearance
        UITabBar.appearance().scrollEdgeAppearance = appearance
    }

    private static func applyNavigationBar() {
        let appearance = UINavigationBarAppearance()
        appearance.configureWithOpaqueBackground()
        appearance.backgroundColor = UIColor(Palette.ink)
        appearance.shadowColor = nil
        let white = UIColor(Palette.text)
        appearance.titleTextAttributes = [
            .foregroundColor: white,
            .font: scaledFont(FontName.bold, size: 17, style: .headline),
        ]
        appearance.largeTitleTextAttributes = [
            .foregroundColor: white,
            .font: scaledFont(FontName.display, size: 34, style: .largeTitle),
        ]
        UINavigationBar.appearance().standardAppearance = appearance
        UINavigationBar.appearance().compactAppearance = appearance
        UINavigationBar.appearance().scrollEdgeAppearance = appearance
    }

    /// A bundled font scaled for the current Dynamic Type size; the system font if it's missing.
    private static func scaledFont(_ name: String, size: CGFloat, style: UIFont.TextStyle) -> UIFont {
        let base = UIFont(name: name, size: size) ?? UIFont.systemFont(ofSize: size, weight: .bold)
        return UIFontMetrics(forTextStyle: style).scaledFont(for: base)
    }

    /// A 1pt-wide strip the bar stretches into its top rule.
    private static func rule(height: CGFloat, color: UIColor) -> UIImage {
        let size = CGSize(width: 1, height: height)
        return UIGraphicsImageRenderer(size: size).image { context in
            color.setFill()
            context.fill(CGRect(origin: .zero, size: size))
        }
    }
}
