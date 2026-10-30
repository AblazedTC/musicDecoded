package com.musicdecoded.mobile.ui.components

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.ui.draw.clip
import androidx.compose.ui.semantics.Role
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.defaultMinSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.Favorite
import androidx.compose.material.icons.rounded.FileUpload
import androidx.compose.material.icons.rounded.MoreHoriz
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.tooling.preview.Preview
import com.musicdecoded.mobile.ui.theme.MusicDecodedColors
import com.musicdecoded.mobile.ui.theme.Shapes
import com.musicdecoded.mobile.ui.theme.Sizes
import com.musicdecoded.mobile.ui.theme.Spacing

/**
 * Round icon button. `active` = primary tint on `primarySoft` (e.g. favorite on, loop on).
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
) {
    Box(
        modifier
            .size(Sizes.minTouchTarget)
            .clip(CircleShape)
            .clickable(role = Role.Button, onClick = onClick),
        contentAlignment = Alignment.Center,
    ) {
        Box(
            Modifier
                .size(if (small) Sizes.smallIconButton else Sizes.iconButton)
                .background(if (active) MusicDecodedColors.PrimarySoft else MusicDecodedColors.Surface, CircleShape)
                .border(Sizes.hairline, if (active) MusicDecodedColors.Primary else MusicDecodedColors.Line, CircleShape),
            contentAlignment = Alignment.Center,
        ) {
            Icon(
                icon,
                contentDescription = contentDescription,
                modifier = Modifier.size(if (small) Sizes.smallIcon else Sizes.icon),
                tint = if (active) MusicDecodedColors.Primary else MusicDecodedColors.TextSecondary,
            )
        }
    }
}

/** Full-width primary action (48 tall, `primary` fill). [loading] shows a spinner and disables it. */
@Composable
fun PrimaryButton(
    text: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    icon: ImageVector? = null,
    enabled: Boolean = true,
    loading: Boolean = false,
) {
    Button(
        onClick = onClick,
        enabled = enabled && !loading,
        modifier = modifier.fillMaxWidth().defaultMinSize(minHeight = Sizes.buttonHeight),
        shape = Shapes.button,
        colors = ButtonDefaults.buttonColors(
            containerColor = MusicDecodedColors.Primary,
            contentColor = MusicDecodedColors.OnPrimary,
            disabledContainerColor = MusicDecodedColors.SurfaceSoft,
            disabledContentColor = MusicDecodedColors.TextMuted,
        ),
    ) {
        if (loading) {
            CircularProgressIndicator(Modifier.size(Sizes.smallIcon), color = MusicDecodedColors.Primary, strokeWidth = Sizes.progressTrack / 2)
        } else {
            ButtonContent(text, icon)
        }
    }
}

/** Full-width secondary action (`surface` fill, `line` border). */
@Composable
fun SecondaryButton(
    text: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    icon: ImageVector? = null,
    enabled: Boolean = true,
) {
    OutlinedButton(
        onClick = onClick,
        enabled = enabled,
        modifier = modifier.fillMaxWidth().defaultMinSize(minHeight = Sizes.buttonHeight),
        shape = Shapes.button,
        border = BorderStroke(Sizes.hairline, MusicDecodedColors.Line),
        colors = ButtonDefaults.outlinedButtonColors(
            containerColor = MusicDecodedColors.Surface,
            contentColor = MusicDecodedColors.Text,
        ),
    ) { ButtonContent(text, icon) }
}

@Composable
private fun ButtonContent(text: String, icon: ImageVector?) {
    Row(horizontalArrangement = Arrangement.spacedBy(Spacing.xs), verticalAlignment = Alignment.CenterVertically) {
        if (icon != null) Icon(icon, contentDescription = null, modifier = Modifier.size(Sizes.smallIcon))
        Text(text, style = MaterialTheme.typography.titleSmall)
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
