package com.musicdecoded.mobile.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.ui.semantics.Role
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.defaultMinSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.size
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.Favorite
import androidx.compose.material.icons.rounded.FileUpload
import androidx.compose.material.icons.rounded.MoreHoriz
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.foundation.layout.padding
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import com.musicdecoded.mobile.ui.theme.Borders
import com.musicdecoded.mobile.ui.theme.Shadows
import androidx.compose.ui.tooling.preview.Preview
import com.musicdecoded.mobile.ui.theme.MusicDecodedColors
import com.musicdecoded.mobile.ui.theme.Sizes
import com.musicdecoded.mobile.ui.theme.Spacing

/**
 * Square icon button. Default = `surface` fill with a white-20% border (on ink); [active] = lime
 * fill with an ink icon (favorite on, loop on, play). [container] overrides the fill for icon
 * buttons on color blocks (e.g. white on violet), which then get a black border.
 * The visual is 40 (or 32 when [small]); the touch target is always ≥ 48dp.
 */
@Composable
fun MdIconButton(
    icon: ImageVector,
    contentDescription: String?,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    active: Boolean = false,
    small: Boolean = false,
    container: Color? = null,
) {
    val fill = container ?: if (active) MusicDecodedColors.Primary else MusicDecodedColors.Surface
    val onColor = active || container != null
    Box(
        modifier
            .size(Sizes.minTouchTarget)
            .clickable(role = Role.Button, onClick = onClick),
        contentAlignment = Alignment.Center,
    ) {
        Box(
            Modifier
                .size(if (small) Sizes.smallIconButton else Sizes.iconButton)
                .background(fill)
                .border(Borders.thick, if (onColor) MusicDecodedColors.Border else MusicDecodedColors.BorderOnInk),
            contentAlignment = Alignment.Center,
        ) {
            Icon(
                icon,
                contentDescription = contentDescription,
                modifier = Modifier.size(if (small) Sizes.smallIcon else Sizes.icon),
                tint = if (onColor) MusicDecodedColors.OnColor else MusicDecodedColors.Text,
            )
        }
    }
}

/**
 * Full-width primary action (48 tall): lime fill, ink text, 3pt black border and a hard shadow
 * (violet on ink by default; pass black on color blocks). [loading] shows a spinner and disables it.
 */
@Composable
fun PrimaryButton(
    text: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    icon: ImageVector? = null,
    enabled: Boolean = true,
    loading: Boolean = false,
    fill: Color = MusicDecodedColors.Primary,
    shadowColor: Color = MusicDecodedColors.Violet,
) {
    val active = enabled && !loading
    WrappedButton(
        text, icon, onClick, modifier, active,
        fill = if (active) fill else MusicDecodedColors.SurfaceRaised,
        content = if (active) MusicDecodedColors.OnColor else MusicDecodedColors.TextMuted,
        border = if (active) MusicDecodedColors.Border else MusicDecodedColors.Line,
        shadow = if (active) Shadows.md else 0.dp,
        shadowColor = shadowColor,
        loading = loading,
    )
}

/** Full-width secondary action: `surface` fill, white text, white-20% border, no shadow. */
@Composable
fun SecondaryButton(
    text: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    icon: ImageVector? = null,
    enabled: Boolean = true,
) {
    WrappedButton(
        text, icon, onClick, modifier, enabled,
        fill = MusicDecodedColors.Surface,
        content = if (enabled) MusicDecodedColors.Text else MusicDecodedColors.TextMuted,
        border = MusicDecodedColors.BorderOnInk,
        shadow = 0.dp,
        shadowColor = MusicDecodedColors.Border,
    )
}

@Composable
private fun WrappedButton(
    text: String,
    icon: ImageVector?,
    onClick: () -> Unit,
    modifier: Modifier,
    enabled: Boolean,
    fill: Color,
    content: Color,
    border: Color,
    shadow: Dp,
    shadowColor: Color,
    loading: Boolean = false,
) {
    Box(
        modifier
            .fillMaxWidth()
            .defaultMinSize(minHeight = Sizes.buttonHeight)
            .block(fill, shadow, shadowColor, border, onClick = onClick, enabled = enabled)
            .padding(horizontal = Spacing.md, vertical = Spacing.xs),
        contentAlignment = Alignment.Center,
    ) {
        if (loading) {
            CircularProgressIndicator(Modifier.size(Sizes.smallIcon), color = MusicDecodedColors.Primary, strokeWidth = Sizes.progressTrack / 2)
        } else {
            Row(horizontalArrangement = Arrangement.spacedBy(Spacing.xs), verticalAlignment = Alignment.CenterVertically) {
                if (icon != null) Icon(icon, contentDescription = null, tint = content, modifier = Modifier.size(Sizes.smallIcon))
                Text(text, style = MaterialTheme.typography.titleSmall, color = content)
            }
        }
    }
}

@Preview
@Composable
private fun IconButtonPreview() = PreviewSurface {
    Row(horizontalArrangement = Arrangement.spacedBy(Spacing.xs)) {
        MdIconButton(Icons.Rounded.Favorite, "Favorite", onClick = {})
        MdIconButton(Icons.Rounded.Favorite, "Favorite", onClick = {}, active = true)
        MdIconButton(Icons.Rounded.MoreHoriz, "More", onClick = {}, small = true)
    }
}

@Preview
@Composable
private fun ButtonsPreview() = PreviewSurface {
    Column(verticalArrangement = Arrangement.spacedBy(Spacing.sm)) {
        PrimaryButton("Import & decode", onClick = {}, icon = Icons.Rounded.FileUpload)
        SecondaryButton("Choose another file", onClick = {})
        PrimaryButton("Decode this song", onClick = {}, enabled = false)
    }
}
