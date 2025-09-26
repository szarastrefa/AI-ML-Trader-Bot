FROM node:18-alpine AS builder

# Set metadata
LABEL maintainer="szarastrefa"
LABEL description="AI/ML Trading Bot React Frontend"
LABEL version="1.0.0"

# Set working directory
WORKDIR /app

# Install dependencies for building
RUN apk add --no-cache git python3 make g++

# Copy package files
COPY frontend/package*.json ./

# Install dependencies (use npm install instead of npm ci since lockfile may not exist)
RUN npm install --only=production && npm cache clean --force

# Copy source code
COPY frontend/ .

# Build the React app
RUN npm run build

# Production stage
FROM nginx:alpine

# Copy built app from builder stage
COPY --from=builder /app/build /usr/share/nginx/html

# Copy custom nginx configuration
COPY docker/nginx.conf /etc/nginx/nginx.conf

# Create nginx user and set permissions
RUN addgroup -g 1001 -S nginx-user && \
    adduser -S -D -H -u 1001 -h /var/cache/nginx -s /sbin/nologin -G nginx-user -g nginx-user nginx-user && \
    chown -R nginx-user:nginx-user /usr/share/nginx/html /var/cache/nginx /var/run /var/log/nginx

# Switch to non-root user
USER nginx-user

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:3000 || exit 1

# Start nginx
CMD ["nginx", "-g", "daemon off;"]