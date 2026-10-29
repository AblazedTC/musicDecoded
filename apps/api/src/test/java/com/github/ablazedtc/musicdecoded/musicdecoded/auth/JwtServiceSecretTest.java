package com.github.ablazedtc.musicdecoded.musicdecoded.auth;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.ConfigDataApplicationContextInitializer;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;
import org.springframework.context.support.PropertySourcesPlaceholderConfigurer;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Loads the real application(-dev).properties with only JwtService, so no database is needed.
 */
class JwtServiceSecretTest {

    private static final String VALID_SECRET = "a-test-secret-that-is-at-least-32-bytes-long";

    private final ApplicationContextRunner runner = new ApplicationContextRunner()
            .withInitializer(new ConfigDataApplicationContextInitializer())
            .withBean(PropertySourcesPlaceholderConfigurer.class)
            .withBean(JwtService.class);

    @BeforeAll
    static void requireNoJwtSecretInEnvironment() {
        assumeTrue(System.getenv("JWT_SECRET") == null, "JWT_SECRET is set in this environment");
    }

    @Test
    void defaultProfileFailsWithoutJwtSecret() {
        runner.run(context -> {
            assertThat(context).hasFailed();
            assertThat(context.getStartupFailure()).rootCause().hasMessageContaining("JWT_SECRET");
        });
    }

    @Test
    void defaultProfileFailsWithBlankJwtSecret() {
        runner.withPropertyValues("JWT_SECRET= ").run(context -> {
            assertThat(context).hasFailed();
            assertThat(context.getStartupFailure()).rootCause().hasMessageContaining("JWT_SECRET is not set");
        });
    }

    @Test
    void defaultProfileFailsWithShortJwtSecret() {
        runner.withPropertyValues("JWT_SECRET=too-short").run(context -> {
            assertThat(context).hasFailed();
            assertThat(context.getStartupFailure()).rootCause().hasMessageContaining("too short");
        });
    }

    @Test
    void defaultProfileStartsWithJwtSecret() {
        runner.withPropertyValues("JWT_SECRET=" + VALID_SECRET).run(context -> {
            assertThat(context).hasNotFailed();
            JwtService jwtService = context.getBean(JwtService.class);
            String token = jwtService.generateToken("a@b.com");
            assertThat(jwtService.extractEmail(token)).isEqualTo("a@b.com");
        });
    }

    @Test
    void devProfileStartsWithoutJwtSecret() {
        runner.withPropertyValues("spring.profiles.active=dev")
                .run(context -> assertThat(context).hasNotFailed());
    }

    @Test
    void constructorRejectsShortSecret() {
        assertThatThrownBy(() -> new JwtService("x".repeat(JwtService.MIN_SECRET_BYTES - 1), 1000))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("at least 32 bytes");
    }
}
