package com.github.ablazedtc.musicdecoded.musicdecoded.common;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;

@RestControllerAdvice
public class ApiExceptionHandler {

    public record ApiError(String code, String message) {}

    @ExceptionHandler(ApiException.class)
    public ResponseEntity<ApiError> handle(ApiException e) {
        return ResponseEntity.status(e.getStatus()).body(new ApiError(e.getCode(), e.getMessage()));
    }
}
